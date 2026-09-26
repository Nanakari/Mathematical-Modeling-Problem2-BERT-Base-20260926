"""Offline contract tests for the v5 student package exporter."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, BertConfig

import v5_export
from v5_data import encode_visible_text, load_tokenizers
from v5_model import MODEL_DEFAULTS, V5Model


def _make_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    backbone = root / "tiny_backbone"
    backbone.mkdir()
    # Keep the source IDs for the four BERT specials consistent with the
    # problem's 0/100/101/102 token convention.
    vocabulary = ["[PAD]"] + [f"[unused{i}]" for i in range(1, 100)]
    vocabulary += ["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocabulary += [f"word{i}" for i in range(104, 120)]
    # These valid WordPiece entries look like credential metadata keys. They
    # must remain untouched inside tokenizer.json and keep their exact IDs.
    vocabulary += ["secret", "token", "authorization", "password", "credentials"]
    (backbone / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
               num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
               pad_token_id=0, type_vocab_size=2).save_pretrained(backbone)
    # AutoTokenizer uses the local BERT config and vocabulary; this file keeps
    # tokenizer construction independent of any model hub lookup.
    (backbone / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True, "model_max_length": 64}, indent=2), encoding="utf-8")

    model_cfg = dict(MODEL_DEFAULTS)
    model_cfg.update({"backbone_dir": str(backbone), "load_pretrained": False,
                      "fusion_dim": 8, "temporal_width": 8, "num_heads": 2,
                      "dropout": 0.0, "teacher_feature_dim": 24,
                      "audio_encoder": "baseline", "vision_encoder": "baseline",
                      "pooling": "masked_mean", "fusion": "pooled"})
    torch.manual_seed(42)
    model = V5Model(backbone, model_cfg, with_distill_heads=True).cpu().eval()
    checkpoint = root / "student.safetensors"
    # Exercise removal of training-only KD projections during package export.
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
              str(checkpoint))
    scaler = root / "standardizer.json"
    scaler.write_text(json.dumps({"fit_split": "train", "stats": {
        "audio": {"count": 8, "mean": [0.0] * 74, "scale": [1.0] * 74},
        "vision": {"count": 8, "mean": [0.0] * 35, "scale": [1.0] * 35},
    }}), encoding="utf-8")
    config = root / "config.json"
    config.write_text(json.dumps({
        "model": model_cfg,
        "source_vocab": str(Path(v5_export.ROOT) / "models" / "bert_mini" / "vocab.txt"),
        "data": {"max_text_length": 16, "feature_clip": 5.0},
    }, indent=2), encoding="utf-8")
    return config, checkpoint, scaler, backbone


def _synthetic_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    text = np.zeros((2, 3, 50), dtype=np.int64)
    audio = np.zeros((2, 50, 74), dtype=np.float32)
    vision = np.zeros((2, 50, 35), dtype=np.float32)
    for row, length in enumerate((8, 11)):
        text[row, 0, 0] = 101
        text[row, 0, 1:length - 1] = rng.integers(104, 120, size=length - 2)
        text[row, 0, 2] = 100  # Source BERT [UNK] exercises target retokenization.
        text[row, 0, length - 1] = 102
        text[row, 1, :length] = 1
        audio[row, :length] = rng.normal(size=(length, 74)).astype(np.float32)
        vision[row, :length] = rng.normal(size=(length, 35)).astype(np.float32)
    return {"text_bert": text, "audio": audio, "vision": vision,
            "id": np.asarray(["a", "b"])}


class V5ExportTests(unittest.TestCase):
    def test_fp16_export_reload_hashes_and_corruption_detection(self):
        with tempfile.TemporaryDirectory(prefix="v5_export_test_") as temp:
            root = Path(temp)
            config, checkpoint, scaler, backbone = _make_fixture(root)
            package = root / "offline_student"
            result = v5_export.export_package(config, checkpoint, scaler, package)

            self.assertTrue(result["package_reload_verified"])
            self.assertLess(result["directory_bytes"], 50_000_000)
            self.assertLess(result["zip_bytes"], 50_000_000)
            self.assertEqual(result["package_sha256"], v5_export.hash_path(package))
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["precision"], "fp16")
            self.assertEqual(manifest["exporter_sha256"], v5_export.sha256_file(v5_export.__file__))
            self.assertTrue(manifest["package_reload_verification"]["verified"])
            self.assertTrue(manifest["tokenizer_strategy"]["target_is_fast"])
            self.assertEqual(manifest["tokenizer_strategy"]["target"],
                             "AutoTokenizer.from_pretrained(tokenizer; local_files_only=true; use_fast=true)")
            self.assertNotIn("distill_adapters", " ".join(manifest["files"]))
            self.assertTrue((package / "backbone" / "config.json").is_file())
            self.assertTrue((package / "source_vocab" / "vocab.txt").is_file())
            self.assertFalse((package / "backbone" / "model.safetensors").exists())

            verification = v5_export.verify_package(package)
            self.assertEqual(verification["status"], "passed")
            self.assertEqual(verification["zip_sha256"], result["zip_sha256"])
            predictor = v5_export.load_offline_predictor(package, device="cpu")
            source_tokenizer = AutoTokenizer.from_pretrained(
                str(backbone), local_files_only=True, use_fast=True)
            self.assertEqual(source_tokenizer.get_vocab(), predictor.tokenizer_target.get_vocab())
            expected_ids = {"secret": 120, "token": 121, "authorization": 122,
                            "password": 123, "credentials": 124}
            for token, token_id in expected_ids.items():
                self.assertEqual(source_tokenizer.get_vocab()[token], token_id)
                self.assertEqual(predictor.tokenizer_target.get_vocab()[token], token_id)
            predictions = predictor.predict_arrays(_synthetic_arrays(), batch_size=2)
            self.assertEqual(predictions["logits"].shape, (2, 3))
            self.assertEqual(predictions["intensity_raw"].shape, (2,))

            with (package / "README.md").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "SHA256/byte validation failed"):
                v5_export.verify_package(package)

    def test_size_gate_and_existing_package_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="v5_export_size_test_") as temp:
            root = Path(temp)
            config, checkpoint, scaler, _ = _make_fixture(root)
            rejected = root / "too_small_limit"
            with self.assertRaisesRegex(ValueError, "limit"):
                v5_export.export_package(config, checkpoint, scaler, rejected,
                                         max_bytes=2_000, verify_process=False)
            self.assertFalse(rejected.exists())
            self.assertFalse(Path(str(rejected) + ".zip").exists())

            complete = root / "already_complete"
            complete.mkdir()
            with self.assertRaises(FileExistsError):
                v5_export.export_package(config, checkpoint, scaler, complete,
                                         verify_process=False)

    def test_int8_is_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory(prefix="v5_export_int8_test_") as temp:
            root = Path(temp)
            config, checkpoint, scaler, _ = _make_fixture(root)
            with self.assertRaisesRegex(NotImplementedError, "INT8 export is unsupported"):
                v5_export.export_package(config, checkpoint, scaler, root / "int8",
                                         precision="int8", verify_process=False)

    def test_fast_slow_tokenizer_parity_for_mini_tiny_and_gap_unk(self):
        project = Path(v5_export.ROOT)
        source_vocab = project / "models" / "bert_mini" / "vocab.txt"
        source, _ = load_tokenizers(project / "models" / "bert_mini", source_vocab=source_vocab)
        source_ids = source.get_vocab()
        ids = np.zeros(50, dtype=np.int64)
        ids[:7] = [source.cls_token_id, source_ids["hello"], source.unk_token_id,
                   source_ids["world"], source_ids["this"], source_ids["movie"], source.sep_token_id]
        attention = np.zeros(50, dtype=bool)
        attention[:7] = True
        observed = attention.copy()
        observed[4] = False  # Leave a gap between visible token islands.
        sample = {
            "input_ids": ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros(50, dtype=np.int64),
            "text_observed": observed,
            "text_pool_mask": observed & ~np.isin(ids, [source.cls_token_id, source.sep_token_id]),
        }

        for model_name in ("bert_mini", "bert_tiny"):
            backbone = project / "models" / model_name
            source_tok, fast_tok = load_tokenizers(backbone, source_vocab=source_vocab)
            slow_tok = AutoTokenizer.from_pretrained(str(backbone), local_files_only=True, use_fast=False)
            fast_result = encode_visible_text(sample, source_tok, fast_tok, max_length=16)
            slow_result = encode_visible_text(sample, source_tok, slow_tok, max_length=16)
            self.assertTrue(fast_result["vocab_compatible"], model_name)
            for key in ("input_ids", "attention_mask", "token_type_ids", "text_pool_mask", "source_spans"):
                np.testing.assert_array_equal(fast_result[key], slow_result[key], err_msg=f"{model_name}: {key}")

        # A small, deliberately different target vocab reaches the real visible
        # island retokenization path while retaining a source [UNK] and gap.
        with tempfile.TemporaryDirectory(prefix="v5_gap_unk_tokenizer_") as temp:
            target_dir = Path(temp)
            target_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                            "hello", "world", "this", "movie", "##s"]
            (target_dir / "vocab.txt").write_text("\n".join(target_vocab) + "\n", encoding="utf-8")
            BertConfig(vocab_size=len(target_vocab), hidden_size=16, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=32,
                       max_position_embeddings=64, pad_token_id=0).save_pretrained(target_dir)
            fast_tok = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True, use_fast=True)
            slow_tok = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True, use_fast=False)
            fast_result = encode_visible_text(sample, source, fast_tok, max_length=16)
            slow_result = encode_visible_text(sample, source, slow_tok, max_length=16)
            self.assertFalse(fast_result["vocab_compatible"])
            for key in ("input_ids", "attention_mask", "token_type_ids", "text_pool_mask", "source_spans"):
                np.testing.assert_array_equal(fast_result[key], slow_result[key], err_msg=f"gap/UNK: {key}")


if __name__ == "__main__":
    unittest.main()
