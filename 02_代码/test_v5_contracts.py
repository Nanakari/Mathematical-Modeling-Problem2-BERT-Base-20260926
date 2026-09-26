"""Focused local contracts for v5's model and aligned data path.

These tests use only the checked-in Tiny BERT config/vocabulary and random
weights. They do not download assets or train a full experiment.
"""
import tempfile
import unittest
from pathlib import Path
import json
import pickle
import random
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer
from safetensors.torch import save_file

from aligned_dataset import AlignedDataset
from v5_data import (
    encode_visible_text,
    fit_train_standardizer,
    load_train_valid,
    load_tokenizers,
    make_missing_view,
)
from v5_model import MODEL_DEFAULTS, V5Model
from v5_train import (
    accumulation_weight,
    backward_sample_weighted,
    build_model,
    call_with_rng_preserved,
    distillation_loss,
    evaluate_validation,
    final_test,
    load_official_test_once,
    main as v5_main,
    supervised_losses,
    training_objective,
    rng_state,
    seed_everything,
)
from v5_export import hash_path
from v5_export import export_package as export_offline_package
from v5_export import load_offline_predictor, verify_package
from pipeline import MaskedStandardizer


ROOT = Path(__file__).resolve().parent
TINY = ROOT / "models" / "bert_tiny"


def _model_config(**overrides):
    cfg = {
        "load_pretrained": False,
        "fusion_dim": 16,
        "temporal_width": 16,
        "num_heads": 4,
        "dropout": 0.0,
        "audio_layers": 1,
        "vision_layers": 1,
    }
    cfg.update(overrides)
    return cfg


def _make_batch(batch_size=2, text_length=8, *, all_missing=False):
    g = torch.Generator().manual_seed(91)
    vocab_size = 30522
    ids = torch.randint(200, vocab_size, (batch_size, text_length), generator=g)
    ids[:, 0] = 101
    text_mask = torch.zeros(batch_size, text_length, dtype=torch.bool)
    pool_mask = torch.zeros_like(text_mask)
    segments = torch.zeros(batch_size, text_length, dtype=torch.long)
    if not all_missing:
        text_mask[0, :5] = True
        pool_mask[0, 1:4] = True
        ids[0, 0] = 101
        ids[0, 4] = 102
        segments[0, :5] = torch.tensor([0, 0, 0, 0, 0])
    audio = torch.randn(batch_size, 50, 74, generator=g)
    vision = torch.randn(batch_size, 50, 35, generator=g)
    audio_mask = torch.zeros(batch_size, 50, dtype=torch.bool)
    vision_mask = torch.zeros_like(audio_mask)
    if not all_missing:
        audio_mask[0, :13] = True
        audio_mask[0, 17:25] = True
        vision_mask[0, 2:19] = True
    return {
        "text_ids": ids,
        "text_mask": text_mask,
        "text_pool_mask": pool_mask,
        "text_segments": segments,
        "audio": audio,
        "audio_mask": audio_mask,
        "vision": vision,
        "vision_mask": vision_mask,
    }


def _clone_batch(batch):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _aligned_fixture(split="train", *, n=2):
    tokens = np.zeros((n, 3, 50), dtype=np.float32)
    tokens[:, 0, :6] = [101, 1200, 100, 2200, 3300, 102]
    tokens[:, 1, :6] = 1
    audio = np.ones((n, 50, 74), dtype=np.float32)
    vision = np.ones((n, 50, 35), dtype=np.float32)
    audio[:, 2] = 0.0  # unavailable AV evidence inside the text extent
    vision[:, 4] = 0.0
    audio[:, 6:] = 0.0
    vision[:, 6:] = 0.0
    classes = np.array([1, 2][:n], dtype=np.int64)
    regression = np.array([0.0, 1.0][:n], dtype=np.float32)
    return AlignedDataset({
        "id": [f"{split}_{i}" for i in range(n)],
        "text_bert": tokens,
        "audio": audio,
        "vision": vision,
        "classification_labels": classes,
        "regression_labels": regression,
    }, labeled=True, source=f"attachment2/{split}", special_token_ids=(101, 102))


def _token_sample(token_ids, *, hidden_positions=()):
    ids = np.zeros(50, dtype=np.int64)
    ids[:len(token_ids)] = token_ids
    attention = np.zeros(50, dtype=bool)
    attention[:len(token_ids)] = True
    observed = attention.copy()
    pool = attention & ~np.isin(ids, [101, 102])
    for pos in hidden_positions:
        attention[pos] = False
        observed[pos] = False
        pool[pos] = False
        ids[pos] = 0
    return {
        "sample_id": "synthetic-text",
        "input_ids": ids,
        "attention_mask": attention,
        "token_type_ids": np.zeros(50, dtype=np.int64),
        "text_observed": observed,
        "text_pool_mask": pool,
    }


class _SplitMap(dict):
    """Raise if a loader tries to index the official test split."""
    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("train/valid preparation indexed the official test split")
        return super().__getitem__(key)


class V5DataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BertTokenizer(vocab_file=str(TINY / "vocab.txt"), do_lower_case=True)
        # A separate tokenizer object exercises the same-vocabulary check.
        cls.target = BertTokenizer(vocab_file=str(TINY / "vocab.txt"), do_lower_case=True)

    def test_masked_unk_remains_observed_when_retokenized(self):
        sample = _token_sample([101, self.source.unk_token_id, 102])
        encoded = encode_visible_text(sample, self.source, self.target, max_length=12,
                                      vocab_compatible=False)
        unk = self.target.unk_token_id
        positions = np.flatnonzero(encoded["input_ids"] == unk)
        self.assertEqual(len(positions), 1)
        self.assertTrue(encoded["text_pool_mask"][positions[0]],
                        "a visible source [UNK] is content, not a missing-token marker")
        np.testing.assert_array_equal(encoded["source_spans"][positions[0]], [1, 2])

    def test_retokenization_preserves_gaps_between_visible_wordpiece_islands(self):
        pieces = self.source.tokenize("electroencephalographically")
        self.assertGreaterEqual(len(pieces), 4, "checked-in BERT vocab should provide a split-word fixture")
        ids = [self.source.cls_token_id] + self.source.convert_tokens_to_ids(pieces) + [self.source.sep_token_id]
        hole = 1 + len(pieces) // 2
        sample = _token_sample(ids, hidden_positions=(hole,))
        encoded = encode_visible_text(sample, self.source, self.target, max_length=32,
                                      vocab_compatible=False)
        internal_gaps = np.flatnonzero(
            (encoded["input_ids"] == self.target.sep_token_id) & ~encoded["text_pool_mask"])
        self.assertGreaterEqual(len(internal_gaps), 2,
                                "separate visible islands need a separator as well as final SEP")
        spans = encoded["source_spans"]
        for start, end in spans[encoded["text_pool_mask"]]:
            self.assertFalse(start <= hole < end,
                             "a retokenized visible token must not span across a hidden source token")
        self.assertTrue(any(np.array_equal(spans[pos], [hole, hole + 1]) for pos in internal_gaps),
                        "the inserted non-content separator should record the missing source gap")

    def test_fast_target_tokenizer_maps_outer_specials_and_visible_unk(self):
        source, target = load_tokenizers(TINY, source_vocab=TINY / "vocab.txt")
        self.assertTrue(target.is_fast)
        pieces = source.tokenize("electroencephalographically")
        self.assertGreaterEqual(len(pieces), 4)
        content = [source.unk_token_id] + source.convert_tokens_to_ids(pieces)
        ids = [source.cls_token_id] + content + [source.sep_token_id]
        hole = 2 + len(pieces) // 2
        sample = _token_sample(ids, hidden_positions=(hole,))
        encoded = encode_visible_text(sample, source, target, max_length=32,
                                      vocab_compatible=False)
        unk_positions = np.flatnonzero(encoded["input_ids"] == target.unk_token_id)
        self.assertTrue(any(encoded["text_pool_mask"][pos]
                            and np.array_equal(encoded["source_spans"][pos], [1, 2])
                            for pos in unk_positions))
        gap_separators = np.flatnonzero(
            (encoded["input_ids"] == target.sep_token_id) & ~encoded["text_pool_mask"])
        self.assertTrue(any(np.array_equal(encoded["source_spans"][pos], [hole, hole + 1])
                            for pos in gap_separators))
        attended = int(encoded["attention_mask"].sum())
        self.assertTrue(encoded["attention_mask"][:attended].all())
        self.assertFalse(encoded["attention_mask"][attended:].any())

    def test_same_wordpiece_path_copies_tokens_and_honors_max_length(self):
        pieces = self.source.tokenize("electroencephalographically")
        ids = [self.source.cls_token_id] + self.source.convert_tokens_to_ids(pieces) + [self.source.sep_token_id]
        sample = _token_sample(ids)
        encoded = encode_visible_text(sample, self.source, self.target, max_length=5,
                                      vocab_compatible=True)
        self.assertTrue(encoded["vocab_compatible"])
        self.assertEqual(len(encoded["input_ids"]), 5)
        self.assertEqual(int(encoded["input_ids"][0]), self.source.cls_token_id)
        self.assertEqual(int(encoded["input_ids"][-1]), self.source.sep_token_id)
        self.assertTrue(encoded["text_pool_mask"][1:-1].any())
        self.assertFalse(encoded["text_pool_mask"][0])
        self.assertFalse(encoded["text_pool_mask"][-1])

    def test_candidate_extent_padding_and_modality_missing_stay_distinct(self):
        dataset = _aligned_fixture(n=1)
        sample = dataset[0]
        self.assertTrue(sample["extent_mask"][2])
        self.assertFalse(sample["audio_observed"][2])
        self.assertFalse(sample["vision_observed"][4])
        self.assertFalse(sample["extent_mask"][6])
        self.assertFalse(sample["audio_observed"][6])

        masked = make_missing_view(sample, ("text",), 0.5, "middle", 77)
        np.testing.assert_array_equal(masked["extent_mask"], sample["extent_mask"])
        np.testing.assert_array_equal(masked["audio_observed"], sample["audio_observed"])
        np.testing.assert_array_equal(masked["vision_observed"], sample["vision_observed"])
        np.testing.assert_array_equal(masked["audio"], sample["audio"])
        np.testing.assert_array_equal(masked["vision"], sample["vision"])
        self.assertTrue(np.all(masked["input_ids"][~masked["text_observed"]] == 0))

    def test_train_valid_loader_discards_test_without_indexing_it(self):
        raw = _SplitMap({
            "train": _aligned_fixture("train").part,
            "valid": _aligned_fixture("valid").part,
            "test": object(),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pkl"
            path.touch()
            train, valid = load_train_valid(path, pickle_loader=lambda _handle: raw,
                                            strict_sizes=False)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(valid), 2)
        self.assertNotIn("test", raw)
        self.assertEqual(train.source, "attachment2/train")
        with self.assertRaises(ValueError):
            fit_train_standardizer(valid)


class V5ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(314159)

    def _model(self, **overrides):
        return V5Model(TINY, _model_config(**overrides))

    def test_masked_input_perturbations_cannot_change_predictions(self):
        model = self._model(fusion="cross_attention", pooling="masked_attention").eval()
        batch = _make_batch()
        before = model(batch)
        changed = _clone_batch(batch)
        changed["text_ids"][~changed["text_mask"]] = 999999
        changed["text_segments"][~changed["text_mask"]] = 77
        changed["audio"][~changed["audio_mask"]] = float("nan")
        changed["vision"][~changed["vision_mask"]] = float("nan")
        after = model(changed)
        for key in ("logits", "intensity_raw", "neutral_logit"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        for group in ("pooled_features", "sequence_features", "distill_features"):
            for name, expected in before[group].items():
                torch.testing.assert_close(expected, after[group][name], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(after["logits"]).all())

    def test_encoder_pooling_and_fusion_matrix_runs_forward_backward(self):
        variants = [
            ("baseline", "baseline", "masked_mean", "pooled", True),
            ("dilated_tcn", "transformer", "masked_attention", "cross_attention", False),
            ("transformer", "conformer", "masked_mean", "cross_attention", True),
            ("conformer", "dilated_tcn", "masked_attention", "pooled", False),
        ]
        batch = _make_batch()
        empty = _make_batch(all_missing=True)
        for audio_mode, vision_mode, pooling, fusion, av_direct in variants:
            with self.subTest(audio=audio_mode, vision=vision_mode, pooling=pooling,
                              fusion=fusion, av_direct=av_direct):
                model = self._model(audio_encoder=audio_mode, vision_encoder=vision_mode,
                                    pooling=pooling, fusion=fusion, av_direct=av_direct)
                output = model(batch)
                self.assertEqual(tuple(output["logits"].shape), (2, 3))
                self.assertEqual(tuple(output["intensity_raw"].shape), (2,))
                self.assertEqual(tuple(output["neutral_logit"].shape), (2,))
                for name in ("text", "audio", "vision", "fused"):
                    self.assertTrue(torch.isfinite(output["pooled_features"][name]).all())
                    self.assertTrue(torch.isfinite(output["distill_features"][name]).all())
                expected_text = model.text_pool(output["sequence_features"]["text"],
                                                batch["text_pool_mask"])
                torch.testing.assert_close(output["pooled_features"]["text"], expected_text)

                all_empty_output = model(empty)
                for value in [all_empty_output["logits"], all_empty_output["intensity_raw"],
                              *all_empty_output["pooled_features"].values(),
                              *all_empty_output["sequence_features"].values(),
                              *all_empty_output["distill_features"].values()]:
                    self.assertTrue(torch.isfinite(value).all())

                loss = (output["logits"].square().mean()
                        + output["intensity_raw"].square().mean()
                        + output["neutral_logit"].square().mean()
                        + sum(value.square().mean() for value in output["pooled_features"].values()))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_empty_text_preserves_audio_and_empty_source_has_no_cross_context(self):
        model = self._model(fusion="cross_attention").eval()
        batch = _make_batch(batch_size=1)
        batch["text_mask"][:] = False
        batch["text_pool_mask"][:] = False
        with torch.no_grad():
            with_av = model(batch)
            self.assertGreater(float(with_av["pooled_features"]["audio"].abs().sum()), 0.0)
            self.assertGreater(float(with_av["pooled_features"]["vision"].abs().sum()), 0.0)
            no_av = _clone_batch(batch)
            no_av["audio_mask"][:] = False
            no_av["vision_mask"][:] = False
            without_av = model(no_av)
        self.assertGreater(float((with_av["logits"] - without_av["logits"]).abs().max()), 1e-7)
        self.assertTrue(torch.isfinite(without_av["logits"]).all())

        query = torch.randn(1, 3, model.fusion_dim)
        source = torch.randn(1, 4, model.fusion_dim)
        query_mask = torch.ones(1, 3, dtype=torch.bool)
        source_mask = torch.zeros(1, 4, dtype=torch.bool)
        context = V5Model._cross(query, source, query_mask, source_mask,
                                 nn.MultiheadAttention(model.fusion_dim, 4, batch_first=True))
        self.assertTrue(torch.equal(context, torch.zeros_like(context)),
                        "an all-missing source must not contribute MHA bias as fake context")

    def test_cross_fusion_ignores_special_only_text_as_content(self):
        model = self._model(fusion="cross_attention").eval()
        has_specials = _make_batch(batch_size=1)
        has_specials["text_ids"][:, :4] = torch.tensor([[101, 102, 0, 0]])
        has_specials["text_mask"][:, :4] = torch.tensor([[True, True, False, False]])
        has_specials["text_pool_mask"][:] = False
        no_text = _clone_batch(has_specials)
        no_text["text_mask"][:] = False
        with torch.no_grad():
            special_output = model(has_specials)
            no_text_output = model(no_text)
        for modality in ("audio", "vision"):
            torch.testing.assert_close(special_output["sequence_features"][modality],
                                       no_text_output["sequence_features"][modality],
                                       rtol=0, atol=0)

    def test_intensity_head_exposes_unclamped_raw_value(self):
        model = self._model(av_direct=False).eval()
        with torch.no_grad():
            model.task_head.weight.zero_()
            model.task_head.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 8.0]))
            output = model(_make_batch(batch_size=1))
        self.assertGreater(float(output["intensity_raw"][0]), 3.0)

    def test_distillation_detaches_real_teacher_and_trains_student_projection(self):
        teacher = V5Model(TINY, _model_config(fusion_dim=24),
                          with_distill_heads=False).eval()
        student = V5Model(TINY, _model_config(fusion_dim=16, teacher_feature_dim=24)).train()
        batch = _make_batch()
        batch["class_label"] = torch.tensor([1, 2], dtype=torch.long)
        batch["regression_label"] = torch.tensor([0.0, 0.7], dtype=torch.float32)
        teacher_output = teacher(batch)
        student_output = student(batch)
        losses = training_objective(
            student_output, batch, teacher_output=teacher_output,
            training={"regression_loss_weight": 0.2, "neutral_aux_weight": 0.25},
            distillation={"temperature": 2.0, "feature_components": ["fused"],
                          "feature_weight": 1.0, "logit_weight": 0.5})
        expected_total = (losses["classification"] + 0.2 * losses["regression"]
                          + 0.25 * losses["neutral_aux"] + 0.5 * losses["logit_kd"]
                          + losses["feature_kd"])
        torch.testing.assert_close(losses["total"], expected_total)
        losses["total"].backward()

        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertIsNotNone(student.task_head.weight.grad)
        self.assertGreater(float(student.task_head.weight.grad[:3].abs().sum()), 0.0)
        projection = student.distill_adapters["fused"].weight
        self.assertIsNotNone(projection.grad)
        self.assertTrue(torch.isfinite(projection.grad).all())
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)

    def test_short_final_gradient_accumulation_window_matches_full_batches(self):
        # Five samples split 2+2+1 with accumulation=2; the last update is a
        # one-sample tail and must use its own denominator.
        x = torch.tensor([[1.0], [2.0], [-1.0], [0.5], [3.0]])
        y = torch.tensor([[0.5], [-0.5], [1.0], [0.0], [2.0]])
        accumulated = nn.Linear(1, 1, bias=False)
        reference = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            accumulated.weight.fill_(0.25)
            reference.weight.fill_(0.25)
        optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        batches = [(0, 2), (2, 4), (4, 5)]

        for window in (batches[:2], batches[2:]):
            sizes = [end - start for start, end in window]
            optimizer.zero_grad(set_to_none=True)
            for start, end in window:
                loss = F.mse_loss(accumulated(x[start:end]), y[start:end])
                backward_sample_weighted(loss, end - start, sizes)
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            start, end = window[0][0], window[-1][1]
            F.mse_loss(reference(x[start:end]), y[start:end]).backward()
            reference_optimizer.step()

        torch.testing.assert_close(accumulated.weight, reference.weight, rtol=0, atol=1e-7)
        self.assertEqual(accumulation_weight(1, [1]), 1.0)


class V5TeacherRngContracts(unittest.TestCase):
    def test_teacher_construction_does_not_shift_matched_student_or_rng(self):
        config = {"model": _model_config(backbone_dir=str(TINY), load_pretrained=False)}

        seed_everything(93017)
        no_kd_student = build_model(config, backbone_dir=TINY)
        no_kd_weights = {key: value.detach().clone()
                         for key, value in no_kd_student.state_dict().items()}
        no_kd_rng = rng_state()

        seed_everything(93017)

        def random_teacher_setup():
            random.random()
            np.random.random(19)
            return build_model(config, backbone_dir=TINY, with_distill_heads=False)

        teacher = call_with_rng_preserved(random_teacher_setup)
        self.assertIsNotNone(teacher)
        kd_student = build_model(config, backbone_dir=TINY, teacher_feature_dim=16)
        self.assertEqual(set(no_kd_weights), set(kd_student.state_dict()))
        for key, expected in no_kd_weights.items():
            torch.testing.assert_close(kd_student.state_dict()[key], expected, rtol=0, atol=0)

        kd_rng = rng_state()
        self.assertEqual(no_kd_rng["python"], kd_rng["python"])
        self.assertEqual(no_kd_rng["numpy"][0], kd_rng["numpy"][0])
        np.testing.assert_array_equal(no_kd_rng["numpy"][1], kd_rng["numpy"][1])
        self.assertEqual(no_kd_rng["numpy"][2:], kd_rng["numpy"][2:])
        self.assertTrue(torch.equal(no_kd_rng["torch_cpu"], kd_rng["torch_cpu"]))
        self.assertEqual(len(no_kd_rng["torch_cuda"]), len(kd_rng["torch_cuda"]))
        for expected, actual in zip(no_kd_rng["torch_cuda"], kd_rng["torch_cuda"]):
            self.assertTrue(torch.equal(expected, actual))

        probe = torch.ones(128)
        torch.set_rng_state(no_kd_rng["torch_cpu"])
        no_kd_dropout = F.dropout(probe, p=0.35, training=True)
        torch.set_rng_state(kd_rng["torch_cpu"])
        kd_dropout = F.dropout(probe, p=0.35, training=True)
        torch.testing.assert_close(kd_dropout, no_kd_dropout, rtol=0, atol=0)


class V5FreezeContracts(unittest.TestCase):
    def test_final_test_cannot_be_repeated_by_changing_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "student.zip"
            package.write_bytes(b"synthetic frozen package identity")
            feature_path = root / "features.pkl"
            part = _aligned_fixture("test", n=2).part
            with feature_path.open("wb") as handle:
                pickle.dump({"train": _aligned_fixture("train").part,
                             "valid": _aligned_fixture("valid").part,
                             "test": part}, handle)
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            freeze_path = root / "freeze.json"
            freeze_path.write_text(json.dumps({
                "status": "frozen", "official_test_evaluated": False,
                "package": str(package), "package_sha256": hash_path(package),
            }), encoding="utf-8")

            with self.assertRaises(PermissionError):
                load_official_test_once(feature_path, freeze_path,
                                        root / "premature_result.json")

            class _SyntheticPredictor:
                def predict_samples(self, samples):
                    logits = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                                     (len(samples), 1))
                    intensity = np.zeros(len(samples), dtype=np.float32)
                    return {"logits": logits, "intensity_raw": intensity}

            output_one = root / "test_result_one.json"
            with mock.patch("v5_export.load_offline_predictor", return_value=_SyntheticPredictor()):
                result = final_test(config_path, freeze_path, package,
                                    feature_path=feature_path, output=output_one)
                self.assertTrue(result["official_test_evaluated"])
                self.assertTrue(output_one.is_file())
                with self.assertRaises((FileExistsError, ValueError)):
                    final_test(config_path, freeze_path, package,
                               feature_path=feature_path, output=root / "different_output.json")


class V5CliContracts(unittest.TestCase):
    def test_evaluate_cli_writes_requested_result_file_and_loads_checkpoint_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_output = root / "candidate"
            training_output.mkdir()
            checkpoint = training_output / "best_model.safetensors"
            checkpoint.write_bytes(b"synthetic checkpoint")
            (training_output / "scaler.json").write_text("{}", encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({
                "output": str(root / "different_config_output"),
                "model": {"backbone_dir": str(TINY)},
            }), encoding="utf-8")
            result_path = root / "validation" / "candidate_v4_32.json"
            train_ds, valid_ds = _aligned_fixture("train"), _aligned_fixture("valid")
            metrics = {"accuracy": 0.5, "macro_f1": 0.4, "mae": 0.6}

            with mock.patch("v5_train.load_train_valid", return_value=(train_ds, valid_ds)), \
                    mock.patch("v5_train._load_standardizer") as load_scaler, \
                    mock.patch("v5_train.model_config_from", return_value={"backbone_dir": str(TINY)}), \
                    mock.patch("v5_train.load_tokenizers", return_value=(object(), object())), \
                    mock.patch("v5_train._load_best_model", return_value=object()), \
                    mock.patch("v5_train._evaluate", return_value=(metrics, None, None)):
                code = v5_main([
                    "evaluate", "--config", str(config), "--checkpoint", str(checkpoint),
                    "--output", str(result_path), "--scenario-set", "quick_all_random",
                ])

            self.assertEqual(code, 0)
            load_scaler.assert_called_once_with(training_output / "scaler.json")
            self.assertTrue(result_path.is_file())
            self.assertFalse((result_path / "validation_metrics.json").exists())
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["complete"], metrics)
            self.assertEqual(result["scenario_count"], 4)


class V5ClassWeightContracts(unittest.TestCase):
    def test_optional_neutral_weight_matches_cross_entropy_and_defaults_unchanged(self):
        logits = torch.tensor([[2.0, 0.1, -0.3], [0.1, 1.8, -0.2],
                               [-0.2, 0.3, 1.7]], requires_grad=True)
        batch = {
            "class_label": torch.tensor([0, 1, 2], dtype=torch.long),
            "regression_label": torch.zeros(3, dtype=torch.float32),
        }
        output = {"logits": logits, "intensity_raw": torch.zeros(3), "neutral_logit": None}
        default = supervised_losses(output, batch, regression_weight=0.0, neutral_weight=0.0)
        expected_default = F.cross_entropy(logits.float(), batch["class_label"])
        torch.testing.assert_close(default["classification"], expected_default)

        class_weights = [1.0, 1.5, 1.0]
        weighted = supervised_losses(output, batch, regression_weight=0.0,
                                     neutral_weight=0.0, class_weights=class_weights)
        expected_weighted = F.cross_entropy(
            logits.float(), batch["class_label"],
            weight=torch.tensor(class_weights, dtype=torch.float32))
        torch.testing.assert_close(weighted["classification"], expected_weighted)
        self.assertNotEqual(float(weighted["classification"].detach()),
                            float(default["classification"].detach()))
        weighted["total"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_class_weights_reject_wrong_shape_or_nonpositive_values(self):
        batch = {"class_label": torch.tensor([1]),
                 "regression_label": torch.zeros(1)}
        output = {"logits": torch.zeros(1, 3), "intensity_raw": torch.zeros(1),
                  "neutral_logit": None}
        for weights in ([1.0, 1.5], [1.0, 0.0, 1.0]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                supervised_losses(output, batch, regression_weight=0.0,
                                  neutral_weight=0.0, class_weights=weights)


class V5ExportContracts(unittest.TestCase):
    def test_random_local_student_exports_under_cap_and_reloads_offline(self):
        # This is packaging verification, not training: initialize from the
        # checked-in Mini config and use the same random checkpoint end to end.
        backbone = ROOT / "models" / "bert_mini"
        config = {
            "model": _model_config(
                backbone_dir=str(backbone), load_pretrained=False,
                fusion_dim=32, temporal_width=16, num_heads=4,
                audio_layers=1, vision_layers=1,
            ),
            "data": {"max_text_length": 32},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            checkpoint_path = root / "student.safetensors"
            model = V5Model(backbone, config["model"], with_distill_heads=False).eval()
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in model.state_dict().items()}, str(checkpoint_path))

            normalizer = MaskedStandardizer()
            normalizer.stats = {
                "audio": {"count": 1, "mean": np.zeros(74), "scale": np.ones(74)},
                "vision": {"count": 1, "mean": np.zeros(35), "scale": np.ones(35)},
            }
            scaler_path = root / "scaler.json"
            normalizer.save(scaler_path)
            package_dir = root / "student_package"
            exported = export_offline_package(
                config_path, checkpoint_path, scaler_path, output=package_dir,
                verify_process=True)

            self.assertEqual(exported["status"], "passed")
            self.assertTrue(exported["package_reload_verified"])
            self.assertLess(exported["directory_bytes"], 50_000_000)
            self.assertLess(exported["zip_bytes"], 50_000_000)
            check = verify_package(exported["zip"])
            self.assertEqual(check["status"], "passed")
            self.assertTrue(check["package_reload_verified"])
            self.assertLess(check["zip_uncompressed_bytes"], 50_000_000)

            predictor = load_offline_predictor(exported["zip"], device="cpu")
            prediction = predictor.predict_samples([_aligned_fixture("special", n=1)[0]])
            self.assertEqual(tuple(prediction["logits"].shape), (1, 3))
            self.assertEqual(tuple(prediction["intensity_raw"].shape), (1,))
            self.assertTrue(np.isfinite(prediction["logits"]).all())
            self.assertTrue(np.isfinite(prediction["intensity_raw"]).all())


if __name__ == "__main__":
    unittest.main()
