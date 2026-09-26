"""Local v5 training, validation, checkpoint, export, and freeze CLI.

The module never downloads model weights. Training requires CUDA. CPU is also
supported for synthetic smoke tests and offline package evaluation/prediction.
Only ``final-test`` may expose the official test split to evaluation code, and
it is guarded by a frozen package manifest and a package-hash one-time marker.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, recall_score
from torch import nn
from torch.nn import functional as F

from aligned_dataset import AlignedDataset
from p2 import DATA
from v5_data import (
    EXPECTED_SPLITS, collate_samples, fit_train_standardizer, load_attachment3,
    load_tokenizers, load_train_valid, make_missing_view, split_fingerprint,
)
from v5_model import MODEL_DEFAULTS, V5Model, inference_state_dict, model_parameter_count


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("outputs/teacher_student_v5_20260924")
PACKAGE_MAX_BYTES = 50_000_000
CLASSES = ("Negative", "Neutral", "Positive")
ALL_MODALITIES = ("text", "audio", "vision")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: str | Path) -> dict[str, str]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.name not in {"optimizer.pt"})
    if not files:
        raise ValueError(f"No model files found below {root}")
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def call_with_rng_preserved(function, *args, **kwargs):
    """Call setup code without changing the caller's Python/NumPy/Torch RNGs.

    Teacher construction may randomly initialize an architecture before its
    checkpoint is loaded. That setup must not alter the matched student's
    initialization or dropout stream relative to a no-teacher control.
    """
    state = rng_state()
    try:
        return function(*args, **kwargs)
    finally:
        restore_rng_state(state)


def model_config_from(config: dict) -> dict:
    model_cfg = dict(MODEL_DEFAULTS)
    model_cfg.update(config.get("model", {}))
    if "backbone_dir" not in model_cfg and "backbone_dir" in config:
        model_cfg["backbone_dir"] = config["backbone_dir"]
    if "load_pretrained" not in model_cfg:
        model_cfg["load_pretrained"] = True
    return model_cfg


def source_vocab_from(config: dict) -> Path | None:
    value = config.get("model", {}).get("source_vocab") or config.get("data", {}).get("source_vocab")
    return resolve_project_path(value) if value else None


def build_model(config: dict, *, backbone_dir: str | Path | None = None,
                teacher_feature_dim: int | None = None,
                with_distill_heads: bool = True) -> V5Model:
    model_cfg = model_config_from(config)
    if teacher_feature_dim is not None:
        model_cfg["teacher_feature_dim"] = int(teacher_feature_dim)
    model_dir = backbone_dir or model_cfg.get("backbone_dir")
    if model_dir is None:
        raise ValueError("Set model.backbone_dir to an existing local pretrained model directory")
    return V5Model(resolve_project_path(model_dir), model_cfg,
                   with_distill_heads=with_distill_heads)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def distillation_loss(teacher_out: dict, student_out: dict, temperature: float = 2.0,
                      feature_components=("fused",), *, feature_weight: float = 1.0) -> dict:
    """Frozen-teacher T² KL and cosine consistency over selected pooled features."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_logits = teacher_out["logits"].detach().float()
    student_logits = student_out["logits"].float()
    soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
    log_student = torch.log_softmax(student_logits / temperature, dim=-1)
    logits_kd = F.kl_div(log_student, soft_targets, reduction="batchmean") * (temperature ** 2)
    components = tuple(feature_components)
    feature_terms = []
    for name in components:
        if name not in teacher_out["pooled_features"] or name not in student_out["distill_features"]:
            raise ValueError(f"Unknown/missing feature distillation component: {name}")
        target = teacher_out["pooled_features"][name].detach().float()
        projected = student_out["distill_features"][name].float()
        if target.shape != projected.shape:
            raise ValueError(f"Teacher/student {name} feature shapes differ: {target.shape}/{projected.shape}")
        target = F.normalize(target, p=2, dim=-1, eps=1e-8)
        projected = F.normalize(projected, p=2, dim=-1, eps=1e-8)
        feature_terms.append((1.0 - (target * projected).sum(dim=-1)).mean())
    feature_kd = torch.stack(feature_terms).mean() if feature_terms else logits_kd.new_zeros(())
    return {"logit_kd": logits_kd, "feature_kd": feature_kd,
            "weighted_feature_kd": feature_kd * float(feature_weight)}


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, *,
                         class_weights=None, loss_type: str = "ce",
                         focal_gamma: float = 1.0) -> torch.Tensor:
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=logits.dtype,
                                        device=logits.device).flatten()
        if (weight_tensor.numel() != 3 or not torch.isfinite(weight_tensor).all()
                or torch.any(weight_tensor <= 0)):
            raise ValueError("class_weights must contain three finite positive values in Negative/Neutral/Positive order")
    if loss_type == "ce":
        # Keep the existing PyTorch CE behavior, including weighted-mean
        # normalization by the sum of target class weights.
        return F.cross_entropy(logits, labels, weight=weight_tensor)
    if loss_type != "focal":
        raise ValueError("classification_loss must be 'ce' or 'focal'")
    gamma = float(focal_gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("focal_gamma must be finite and non-negative")
    log_probabilities = F.log_softmax(logits, dim=-1)
    target_log_probabilities = log_probabilities.gather(1, labels.reshape(-1, 1)).squeeze(1)
    target_probabilities = target_log_probabilities.exp()
    per_sample = -torch.pow(1.0 - target_probabilities, gamma) * target_log_probabilities
    if weight_tensor is not None:
        per_sample = per_sample * weight_tensor[labels]
    return per_sample.mean()


def supervised_losses(output: dict, batch: dict, *, regression_weight: float = 0.2,
                      neutral_weight: float = 0.25,
                      class_weights=None, classification_loss: str = "ce",
                      focal_gamma: float = 1.0,
                      consistency_weight: float = 0.0) -> dict:
    labels = batch["class_label"].long()
    regression = batch["regression_label"].float()
    logits = output["logits"].float()
    classification_loss_value = _classification_loss(
        logits, labels, class_weights=class_weights,
        loss_type=classification_loss, focal_gamma=focal_gamma)
    regression_loss = F.smooth_l1_loss(output["intensity_raw"].float(), regression)
    consistency_weight = float(consistency_weight)
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency_weight must be finite and non-negative")
    if consistency_weight > 0:
        if "intensity_direct" not in output or output["intensity_direct"] is None:
            raise ValueError("consistency_weight > 0 requires output['intensity_direct']")
        direct = output["intensity_direct"].float()
        raw = output["intensity_raw"].float()
        if direct.shape != raw.shape:
            raise ValueError(f"intensity_direct/intensity_raw shapes differ: {direct.shape}/{raw.shape}")
        consistency_loss = F.smooth_l1_loss(direct, raw)
    else:
        consistency_loss = classification_loss_value.new_zeros(())
    if output["neutral_logit"] is None:
        if neutral_weight:
            raise ValueError("neutral_aux_weight > 0 requires the auxiliary neutral head")
        neutral_loss = classification_loss_value.new_zeros(())
    else:
        neutral_targets = (labels == 1).to(dtype=torch.float32)
        neutral_loss = F.binary_cross_entropy_with_logits(output["neutral_logit"].float(), neutral_targets)
    total = (classification_loss_value + float(regression_weight) * regression_loss
             + float(neutral_weight) * neutral_loss
             + consistency_weight * consistency_loss)
    return {"total": total, "classification": classification_loss_value,
            "regression": regression_loss, "neutral_aux": neutral_loss,
            "consistency": consistency_loss}


def training_objective(student_output: dict, student_batch: dict, *,
                       teacher_output: dict | None = None,
                       training: dict | None = None,
                       distillation: dict | None = None) -> dict:
    """Shared supervised/KD objective used by CUDA training and CPU smoke."""
    training = training or {}
    distillation = distillation or {}
    losses = supervised_losses(
        student_output, student_batch,
        regression_weight=float(training.get("regression_loss_weight", 0.2)),
        neutral_weight=float(training.get("neutral_aux_weight", 0.25)),
        class_weights=training.get("class_weights"),
        classification_loss=str(training.get("classification_loss", "ce")),
        focal_gamma=float(training.get("focal_gamma", 1.0)),
        consistency_weight=float(training.get("consistency_weight", 0.0)))
    if teacher_output is not None:
        kd = distillation_loss(
            teacher_output, student_output,
            float(distillation.get("temperature", 2.0)),
            tuple(distillation.get("feature_components", ["fused"])),
            feature_weight=float(distillation.get("feature_weight", 0.1)))
        total = (losses["total"]
                 + float(distillation.get("logit_weight", 1.0)) * kd["logit_kd"]
                 + kd["weighted_feature_kd"])
        losses["logit_kd"] = kd["logit_kd"]
        losses["feature_kd"] = kd["feature_kd"]
    else:
        total = losses["total"]
        losses["logit_kd"] = total.new_zeros(())
        losses["feature_kd"] = total.new_zeros(())
    losses["total"] = total
    return losses


def backward_sample_weighted(loss: torch.Tensor, sample_count: int,
                             window_batch_sizes, scaler=None) -> None:
    """Accumulate microbatch-mean losses as one sample-weighted window mean."""
    weight = accumulation_weight(sample_count, window_batch_sizes)
    weighted = loss * weight
    (scaler.scale(weighted) if scaler is not None else weighted).backward()


def accumulation_weight(batch_size: int, window_batch_sizes) -> float:
    """Return the exact sample weight for one batch in an accumulation window."""
    sizes = [int(value) for value in window_batch_sizes]
    if not sizes or int(batch_size) <= 0 or int(batch_size) not in sizes or any(x <= 0 for x in sizes):
        raise ValueError("batch size must be present in a nonempty positive accumulation window")
    return float(batch_size) / float(sum(sizes))


def restore_training_state(state: dict, model, optimizer, scheduler, amp_scaler,
                           *, expected_contract: dict, config_hash: str,
                           source_hashes: dict, model_hashes: dict) -> None:
    """Strictly restore all state needed to continue at the next epoch."""
    if (state.get("config_hash") != config_hash
            or state.get("source_hashes") != source_hashes
            or state.get("model_hashes") != model_hashes
            or state.get("resume_contract") != expected_contract):
        raise ValueError("Resume refused: config, code/model, split, scaler, or teacher fingerprints changed")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    amp_state = state.get("amp_scaler_state")
    if amp_state:
        amp_scaler.load_state_dict(amp_state)
    restore_rng_state(state["rng_state"])


def metric_record(y_true, intensity_true, logits, intensity_raw) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    r = np.asarray(intensity_true, dtype=np.float64)
    p = np.asarray(logits, dtype=np.float64)
    z = np.clip(np.asarray(intensity_raw, dtype=np.float64), -3.0, 3.0)
    prediction = p.argmax(axis=1)
    recalls = recall_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y, prediction, labels=[0, 1, 2])
    if len(y) > 1 and float(np.std(r)) > 0 and float(np.std(z)) > 0:
        pearson = float(np.corrcoef(r, z)[0, 1])
    else:
        pearson = None
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "negative_recall": float(recalls[0]),
        "neutral_recall": float(recalls[1]),
        "positive_recall": float(recalls[2]),
        "neutral_to_positive": int(cm[1, 2]),
        "neutral_correct": int(cm[1, 1]),
        "neutral_count": int(cm[1].sum()),
        "mae": float(mean_absolute_error(r, z)),
        "pearson": pearson,
        "confusion_matrix": cm.tolist(),
    }


def selection_rank(metrics: dict) -> tuple[float, float, float]:
    return (float(metrics["accuracy"]), float(metrics["macro_f1"]), -float(metrics["mae"]))


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_dtype)
    if dtype is None:
        raise ValueError("amp_dtype must be none, bf16, or fp16")
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not report bfloat16 support; set amp_dtype=fp16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _sample_views(samples, indices, *, policy: str, seed: int, epoch: int,
                  rates=(0.1, 0.3, 0.5), position_choices=("start", "middle", "end", "random"),
                  clean_probability: float = 0.6):
    if policy == "full":
        return [samples[int(i)] for i in indices]
    if policy not in {"missing", "mixed"}:
        raise ValueError(f"Unknown view policy {policy}")
    if not 0.0 <= float(clean_probability) <= 1.0:
        raise ValueError("clean_probability must be in [0,1]")
    views = []
    combinations = [("text",), ("audio",), ("vision",), ("text", "audio"),
                    ("text", "vision"), ("audio", "vision"), ALL_MODALITIES]
    for index in indices:
        original = samples[int(index)]
        rng = np.random.default_rng(seed + epoch * 1000003 + int(index) * 97)
        if policy == "mixed" and rng.random() < float(clean_probability):
            views.append(original)
            continue
        modalities = combinations[int(rng.integers(len(combinations)))]
        rate = float(rates[int(rng.integers(len(rates)))])
        position = str(position_choices[int(rng.integers(len(position_choices)))])
        views.append(make_missing_view(original, modalities, rate, position,
                                       int(seed + epoch * 1009 + index), nested_rates=rates))
    return views


def _missing_curriculum_state(training: dict, epoch_index: int) -> dict:
    """Return the effective per-epoch mask rates; epoch_index is zero-based."""
    mode = str(training.get("missing_curriculum", "none"))
    if mode not in {"none", "severity_ramp"}:
        raise ValueError("training.missing_curriculum must be 'none' or 'severity_ramp'")
    rates = tuple(training.get("mask_rates", [0.1, 0.3, 0.5]))
    if mode == "severity_ramp":
        progress = min(1.0, max(0.0, float(epoch_index) / 4.0))
        factor = 0.5 + 0.5 * progress
        effective_rates = tuple(float(rate) * factor for rate in rates)
    else:
        factor = 1.0
        effective_rates = rates
    return {"mode": mode, "factor": float(factor),
            "mask_rates": effective_rates}


def _evaluate(model: V5Model, samples, y, regression, normalizer, source_tokenizer,
              target_tokenizer, *, device, batch_size: int, max_text_length: int,
              amp_dtype="none") -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    logits, intensities = [], []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            current = samples[start:start + batch_size]
            batch = collate_samples(current, normalizer, source_tokenizer, target_tokenizer,
                                    max_text_length, device=device)
            with _autocast_context(device, amp_dtype):
                output = model(batch)
            logits.append(output["logits"].float().cpu().numpy())
            intensities.append(output["intensity_raw"].float().cpu().numpy())
    p = np.concatenate(logits) if logits else np.zeros((0, 3), dtype=np.float32)
    z = np.concatenate(intensities) if intensities else np.zeros(0, dtype=np.float32)
    return metric_record(y, regression, p, z), p, z


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_safetensors(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.safetensors")
    save_file({key: value.detach().cpu().contiguous() for key, value in payload.items()}, str(temporary))
    os.replace(temporary, path)


def prepare(config_path: str | Path, *, output: str | Path | None = None,
            feature_path: str | Path | None = None) -> Path:
    config_path = resolve_project_path(config_path)
    config = read_json(config_path)
    output_path = resolve_project_path(output or config.get("output", DEFAULT_OUTPUT))
    if (output_path / "prepared_manifest.json").exists():
        raise FileExistsError(f"Prepared output already exists: {output_path}; choose a new output directory")
    train, valid = load_train_valid(feature_path)
    normalizer = fit_train_standardizer(train)
    output_path.mkdir(parents=True, exist_ok=True)
    normalizer.save(output_path / "scaler.json")
    model_cfg = model_config_from(config)
    backbone = resolve_project_path(model_cfg["backbone_dir"])
    source_files = hash_tree(backbone)
    source_tok, target_tok = load_tokenizers(backbone, source_vocab=source_vocab_from(config))
    vocab_compatible = source_tok.get_vocab() == target_tok.get_vocab()
    # Tokenize only input-visible text_bert content. No raw_text or cached full
    # encoding enters these records; masks are applied before this boundary.
    length_stats = []
    for dataset in (train, valid):
        for index in range(len(dataset)):
            encoded = __import__("v5_data").encode_visible_text(
                dataset[index], source_tok, target_tok,
                int(config.get("data", {}).get("max_text_length", 128)),
                vocab_compatible=vocab_compatible)
            length_stats.append(int(encoded["attention_mask"].sum()))
    manifest = {
        "train": split_fingerprint(train),
        "valid": split_fingerprint(valid),
        "expected_counts": EXPECTED_SPLITS,
        "official_test_loaded_for_selection": False,
        "train_only_standardizer": True,
        "text_vocab_compatible": bool(vocab_compatible),
        "visible_token_retokenization_only": True,
        "encoded_length_min": int(min(length_stats)),
        "encoded_length_max": int(max(length_stats)),
        "backbone_files": source_files,
        "scaler_sha256": sha256_file(output_path / "scaler.json"),
        "source_hashes": {name: sha256_file(ROOT / name) for name in
                          ("v5_model.py", "v5_data.py", "v5_train.py", "aligned_dataset.py", "pipeline.py")},
        "model_id": config.get("model_id"),
        "model_revision": config.get("model_revision"),
        "protocol_sha256": canonical_hash(config),
    }
    write_json(output_path / "protocol.json", config)
    write_json(output_path / "prepared_manifest.json", manifest)
    write_json(output_path / "source_hashes.json", {
        name: sha256_file(ROOT / name) for name in
        ("v5_model.py", "v5_data.py", "v5_train.py", "aligned_dataset.py", "pipeline.py")
    })
    return output_path


def _load_standardizer(path: str | Path):
    from pipeline import MaskedStandardizer
    return MaskedStandardizer.load(path)


def _load_best_model(config: dict, checkpoint_path: str | Path, *, device,
                     teacher_feature_dim: int | None = None, with_distill_heads=True):
    model = build_model(config, teacher_feature_dim=teacher_feature_dim,
                        with_distill_heads=with_distill_heads).to(device)
    state = load_file(str(checkpoint_path), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.startswith("distill_adapters.")]
    if missing or unexpected:
        raise ValueError(f"Checkpoint/model mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    return model


def evaluate_validation(config_path: str | Path, checkpoint_path: str | Path,
                        *, output: str | Path | None = None,
                        feature_path: str | Path | None = None,
                        scenario_set: str = "v4_32") -> dict:
    config_path = resolve_project_path(config_path)
    config = read_json(config_path)
    checkpoint_file = resolve_project_path(checkpoint_path)
    configured_training_output = resolve_project_path(config.get("output", DEFAULT_OUTPUT))
    # Checkpoints are written beside the train-fitted scaler. This remains
    # correct when `train --output` overrode config.output.
    checkpoint_training_output = checkpoint_file.parent
    scaler_path = (checkpoint_training_output / "scaler.json"
                   if (checkpoint_training_output / "scaler.json").is_file()
                   else configured_training_output / "scaler.json")
    result_path = (resolve_project_path(output) if output is not None
                   else configured_training_output / "validation_metrics.json")
    train_ds, valid_ds = load_train_valid(feature_path)
    normalizer = _load_standardizer(scaler_path)
    model_cfg = model_config_from(config)
    backbone = resolve_project_path(model_cfg["backbone_dir"])
    source_tok, target_tok = load_tokenizers(backbone, source_vocab=source_vocab_from(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_best_model(config, checkpoint_file, device=device,
                             with_distill_heads=False)
    valid = [valid_ds[i] for i in range(len(valid_ds))]
    y = valid_ds.part["classification_labels"].astype(int)
    r = valid_ds.part["regression_labels"].astype(float)
    tr = config.get("training", {})
    if scenario_set not in {"v4_32", "quick_all_random"}:
        raise ValueError("scenario_set must be v4_32 or quick_all_random")
    definitions = validation_case_definitions(int(tr.get("validation_mask_seed", 101)))
    if scenario_set == "quick_all_random":
        definitions = [definitions[0]] + [case for case in definitions
                                          if case["name"] in {"all|random|0.1", "all|random|0.5", "all|random|0.7"}]
    metrics_by_case = {}
    for case in definitions:
        selected = validation_case_samples(valid, case)
        row, _, _ = _evaluate(
            model, selected, y, r, normalizer, source_tok, target_tok,
            device=device, batch_size=int(tr.get("batch_size", 32)),
            max_text_length=int(config.get("data", {}).get("max_text_length", 128)),
            amp_dtype=tr.get("amp_dtype", "none"))
        metrics_by_case[case["name"]] = row
    complete = metrics_by_case.pop("complete")
    missing_mean = (float(np.mean([row["accuracy"] for row in metrics_by_case.values()]))
                    if metrics_by_case else None)
    quick_all_random = {name: metrics_by_case[name] for name in
                        ("all|random|0.1", "all|random|0.5", "all|random|0.7")
                        if name in metrics_by_case}
    result = {
        "complete": complete,
        "missing_scenarios": metrics_by_case,
        "missing_scenario_count": len(metrics_by_case),
        "missing_scenario_mean_accuracy": missing_mean,
        "missing_all_random": quick_all_random,
        "scenario_set": scenario_set,
        "scenario_count": len(metrics_by_case) + 1,
        "selection_rank": list(selection_rank(complete)),
        "official_test_evaluated": False,
        "checkpoint_sha256": sha256_file(checkpoint_file),
    }
    write_json(result_path, result)
    return result


def _training_source_hashes() -> dict[str, str]:
    names = ("v5_model.py", "v5_data.py", "v5_train.py", "aligned_dataset.py", "pipeline.py")
    return {name: sha256_file(ROOT / name) for name in names}


def _make_missing_batch_views(train_samples, indices, seed, epoch, cfg, *, curriculum_state=None):
    training = cfg.get("training", {})
    policy = training.get("view_policy", "full")
    if policy == "full":
        return [train_samples[int(i)] for i in indices]
    curriculum_state = curriculum_state or _missing_curriculum_state(training, epoch)
    return _sample_views(train_samples, indices, policy=policy, seed=seed, epoch=epoch,
                         rates=tuple(curriculum_state["mask_rates"]),
                         clean_probability=float(training.get("clean_probability", 0.6)))


def _save_best(model: V5Model, out: Path, epoch: int, metrics: dict):
    _atomic_safetensors(inference_state_dict(model), out / "best_model.safetensors")
    write_json(out / "best_validation.json", {
        "epoch": int(epoch), "metrics": metrics,
        "rank": list(selection_rank(metrics)), "official_test_evaluated": False,
    })


def train(config_path: str | Path, *, output: str | Path | None = None,
          feature_path: str | Path | None = None, teacher_checkpoint: str | Path | None = None,
          teacher_config_path: str | Path | None = None, resume: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. Use the explicit `smoke` command for CPU checks.")
    config_path = resolve_project_path(config_path)
    config = read_json(config_path)
    out = resolve_project_path(output or config.get("output", DEFAULT_OUTPUT))
    training = config.get("training", {})
    seed = int(training.get("seed", 1729))
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 32))
    accum_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    device = torch.device("cuda")
    seed_everything(seed)
    if not (out / "prepared_manifest.json").exists():
        prepare(config_path, output=out, feature_path=feature_path)
    train_ds, valid_ds = load_train_valid(feature_path)
    prepared_manifest = read_json(out / "prepared_manifest.json")
    current_fingerprints = {
        "train": split_fingerprint(train_ds),
        "valid": split_fingerprint(valid_ds),
    }
    if prepared_manifest.get("train") != current_fingerprints["train"] or prepared_manifest.get("valid") != current_fingerprints["valid"]:
        raise ValueError("Prepared split fingerprints differ from the current fixed train/valid data")
    train_samples = [train_ds[i] for i in range(len(train_ds))]
    valid_samples = [valid_ds[i] for i in range(len(valid_ds))]
    y = valid_ds.part["classification_labels"].astype(int)
    r = valid_ds.part["regression_labels"].astype(float)
    normalizer = _load_standardizer(out / "scaler.json")
    model_cfg = model_config_from(config)
    backbone = resolve_project_path(model_cfg["backbone_dir"])
    current_model_hashes = hash_tree(backbone)
    if prepared_manifest.get("protocol_sha256") != canonical_hash(config):
        raise ValueError("Prepared protocol differs from the current training config")
    if prepared_manifest.get("backbone_files") != current_model_hashes:
        raise ValueError("Pretrained model files changed since prepare")
    if prepared_manifest.get("source_hashes") != _training_source_hashes():
        raise ValueError("Training code changed since prepare; choose a new output directory and prepare again")
    if prepared_manifest.get("scaler_sha256") != sha256_file(out / "scaler.json"):
        raise ValueError("Train-fitted AV scaler changed since prepare")
    source_tok, target_tok = load_tokenizers(backbone, source_vocab=source_vocab_from(config))
    teacher = None
    teacher_feature_dim = None
    distill_cfg = config.get("distillation", {})
    if distill_cfg.get("enabled", False):
        if teacher_checkpoint is None or teacher_config_path is None:
            raise ValueError("Enabled distillation requires --teacher-checkpoint and --teacher-config")

        def initialize_teacher():
            teacher_cfg = read_json(resolve_project_path(teacher_config_path))
            teacher_model_cfg = model_config_from(teacher_cfg)
            teacher_backbone = resolve_project_path(teacher_model_cfg["backbone_dir"])
            teacher_model = build_model(teacher_cfg, backbone_dir=teacher_backbone,
                                       with_distill_heads=False).to(device)
            teacher_state = load_file(str(resolve_project_path(teacher_checkpoint)), device=str(device))
            missing, unexpected = teacher_model.load_state_dict(teacher_state, strict=False)
            if missing or unexpected:
                raise ValueError(f"Teacher checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)
            source_tok, target_tok = load_tokenizers(
                teacher_backbone, source_vocab=source_vocab_from(teacher_cfg))
            teacher_max_text_length = int(teacher_cfg.get("data", {}).get("max_text_length", 128))
            provenance = {
                "checkpoint_sha256": sha256_file(resolve_project_path(teacher_checkpoint)),
                "config_sha256": sha256_file(resolve_project_path(teacher_config_path)),
                "config_canonical_sha256": canonical_hash(teacher_cfg),
                "backbone_files": hash_tree(teacher_backbone),
            }
            return (teacher_model, teacher_model.fusion_dim, source_tok,
                    target_tok, provenance, teacher_max_text_length)

        (teacher, teacher_feature_dim, teacher_source_tok,
         teacher_target_tok, teacher_provenance,
         teacher_max_text_length) = call_with_rng_preserved(initialize_teacher)
    else:
        teacher_source_tok = teacher_target_tok = None
        teacher_provenance = {"enabled": False}
        teacher_max_text_length = None
    model = build_model(config, teacher_feature_dim=teacher_feature_dim).to(device)
    params_encoder, params_head = [], []
    for name, parameter in model.named_parameters():
        (params_encoder if name.startswith("text_encoder.") else params_head).append(parameter)
    encoder_lr = float(training.get("encoder_lr", 2e-5))
    head_lr = float(training.get("head_lr", 5e-4))
    optimizer = torch.optim.AdamW([
        {"params": params_encoder, "lr": encoder_lr},
        {"params": params_head, "lr": head_lr},
    ], weight_decay=float(training.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_dtype = training.get("amp_dtype", "bf16")
    use_scaler = amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    source_hashes = _training_source_hashes()
    model_hashes = current_model_hashes
    config_hash = canonical_hash(config)
    resume_contract = {
        "split_fingerprints": current_fingerprints,
        "scaler_sha256": sha256_file(out / "scaler.json"),
        "teacher": teacher_provenance,
    }
    last_path = out / "last_training_state.pt"
    start_epoch, best_epoch, best_rank, best_metrics, stale = 0, 0, None, None, 0
    history_path = out / "training_history.jsonl"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError("--resume requested but last_training_state.pt does not exist")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        restore_training_state(state, model, optimizer, scheduler, scaler,
                               expected_contract=resume_contract, config_hash=config_hash,
                               source_hashes=source_hashes, model_hashes=model_hashes)
        start_epoch = int(state["completed_epoch"])
        best_epoch, best_rank, best_metrics, stale = state["best_epoch"], state["best_rank"], state["best_metrics"], state["stale_epochs"]
        restore_rng_state(state["rng_state"])
    else:
        if last_path.exists() or (out / "best_model.safetensors").exists():
            raise FileExistsError("Output contains prior training. Use --resume or choose a new output directory.")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "protocol.json", config)
        write_json(out / "pretrained_provenance.json", {
            "model_id": config.get("model_id"), "revision": config.get("model_revision"),
            "files": model_hashes, "local_files_only": True,
        })
    log_path = out / "training_history.jsonl"
    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(seed + epoch * 1000003).permutation(len(train_samples))
        curriculum_state = _missing_curriculum_state(training, epoch)
        running = {name: 0.0 for name in ("total", "classification", "regression", "neutral_aux",
                                          "consistency", "logit_kd", "feature_kd")}
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch_starts = list(range(0, len(order), batch_size))
        for batch_index, start in enumerate(batch_starts):
            idx = order[start:start + batch_size]
            full_samples = [train_samples[int(i)] for i in idx]
            student_samples = _make_missing_batch_views(
                train_samples, idx, seed, epoch, config,
                curriculum_state=curriculum_state)
            student_batch = collate_samples(student_samples, normalizer, source_tok, target_tok,
                                            int(config.get("data", {}).get("max_text_length", 128)),
                                            device=device)
            teacher_output = None
            if teacher is not None:
                teacher_batch = collate_samples(full_samples, normalizer, teacher_source_tok,
                                                teacher_target_tok,
                                                teacher_max_text_length,
                                                device=device)
                with torch.inference_mode(), _autocast_context(device, amp_dtype):
                    teacher_output = teacher(teacher_batch)
            with _autocast_context(device, amp_dtype):
                student_output = model(student_batch)
                losses = training_objective(student_output, student_batch,
                                            teacher_output=teacher_output,
                                            training=training, distillation=distill_cfg)
                total = losses["total"]
            window_start = (batch_index // accum_steps) * accum_steps
            window_end = min(window_start + accum_steps, len(batch_starts))
            window_sizes = [min(batch_size, len(order) - batch_starts[j])
                            for j in range(window_start, window_end)]
            backward_sample_weighted(total, len(idx), window_sizes,
                                     scaler if use_scaler else None)
            final_batch = batch_index + 1 == len(batch_starts)
            if (batch_index + 1) % accum_steps == 0 or final_batch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_n = len(idx)
            for name in running:
                if name in losses:
                    running[name] += float(losses[name].detach().float().item()) * batch_n
            seen += batch_n
        scheduler.step()
        valid_metrics, _, _ = _evaluate(
            model, valid_samples, y, r, normalizer, source_tok, target_tok,
            device=device, batch_size=batch_size,
            max_text_length=int(config.get("data", {}).get("max_text_length", 128)),
            amp_dtype=amp_dtype)
        rank = selection_rank(valid_metrics)
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch, best_metrics, stale = rank, epoch + 1, valid_metrics, 0
            _save_best(model, out, epoch + 1, valid_metrics)
        else:
            stale += 1
        if scheduler is not None:
            scheduler_state = scheduler.state_dict()
        else:
            scheduler_state = {}
        state = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_state, "amp_scaler_state": scaler.state_dict(),
            "completed_epoch": epoch + 1, "best_epoch": best_epoch,
            "best_rank": best_rank, "best_metrics": best_metrics, "stale_epochs": stale,
            "seed": seed, "sampler_state": {"next_epoch_seed": seed + (epoch + 1) * 1000003},
            "rng_state": rng_state(), "config_hash": config_hash,
            "source_hashes": source_hashes, "model_hashes": model_hashes,
            "resume_contract": resume_contract,
        }
        _atomic_torch_save(state, last_path)
        row = {
            "epoch": epoch + 1, "train_loss": {name: value / max(1, seen) for name, value in running.items()},
            "valid": valid_metrics, "best_epoch": best_epoch, "best_rank": list(best_rank),
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "missing_curriculum": {
                "mode": curriculum_state["mode"], "epoch": epoch + 1,
                "factor": curriculum_state["factor"],
                "mask_rates": list(curriculum_state["mask_rates"]),
            },
            "amp_dtype": amp_dtype, "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "official_test_evaluated": False,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale >= int(training.get("patience", 4)):
            break
    if not (out / "best_model.safetensors").exists():
        raise RuntimeError("Training produced no best checkpoint")
    best_model = _load_best_model(config, out / "best_model.safetensors", device=device,
                                  with_distill_heads=False)
    missing_rows = {}
    rates = tuple(training.get("evaluation_mask_rates", [0.3, 0.5, 0.7]))
    for rate in rates:
        masked = [make_missing_view(sample, ALL_MODALITIES, float(rate), "random",
                                    int(training.get("validation_mask_seed", 101)),
                                    nested_rates=tuple(sorted(set([0.1, 0.3, 0.5, float(rate)]))))
                  for sample in valid_samples]
        score, _, _ = _evaluate(
            best_model, masked, y, r, normalizer, source_tok, target_tok,
            device=device, batch_size=batch_size,
            max_text_length=int(config.get("data", {}).get("max_text_length", 128)),
            amp_dtype=amp_dtype)
        missing_rows[str(rate)] = score
    summary = {
        "candidate": config.get("candidate_name", out.name), "seed": seed,
        "best_epoch": best_epoch, "complete": best_metrics,
        "missing_all_random": missing_rows,
        "selection_order": ["complete_accuracy", "macro_f1", "lower_mae"],
        "rank": list(best_rank), "checkpoint": "best_model.safetensors",
        "checkpoint_sha256": sha256_file(out / "best_model.safetensors"),
            "model_parameters": model_parameter_count(best_model),
        "model_config": model_cfg, "resume_contract": resume_contract,
        "official_test_evaluated": False,
    }
    write_json(out / "candidate_summary.json", summary)
    return summary


def validation_case_definitions(seed: int = 101) -> list[dict]:
    """The same 32 frozen diagnostic cases used by temporal_v4: complete + 31 masked."""
    modalities = ("text", "audio", "vision")
    combinations = [combo for size in (1, 2, 3)
                    for combo in itertools.combinations(modalities, size)]
    cases = [{"name": "complete", "modalities": (), "rate": 0.0,
              "position": "start", "seed": 0}]
    for combo, position in itertools.product(combinations, ("start", "middle", "end", "random")):
        cases.append({"name": f"{'+'.join(combo)}|{position}|0.3",
                      "modalities": combo, "rate": 0.3,
                      "position": position, "seed": int(seed)})
    for rate in (0.1, 0.5, 0.7):
        cases.append({"name": f"all|random|{rate}", "modalities": modalities,
                      "rate": rate, "position": "random", "seed": int(seed)})
    if len(cases) != 32 or len({case["name"] for case in cases}) != 32:
        raise AssertionError("The v4 comparable validation suite must have 32 unique cases")
    return cases


def validation_case_samples(samples, case: dict) -> list:
    if case["name"] == "complete":
        return list(samples)
    return [make_missing_view(sample, case["modalities"], float(case["rate"]),
                              str(case["position"]), int(case["seed"]),
                              nested_rates=(0.1, 0.3, 0.5, 0.7))
            for sample in samples]


def run_smoke(model_dir: str | Path, *, output: str | Path = DEFAULT_OUTPUT,
              device_name: str = "cpu") -> dict:
    """One synthetic forward/backward/save/reload/mask-leak check; no data access."""
    model_dir = resolve_project_path(model_dir)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    seed_everything(7081)
    cfg = {
        **MODEL_DEFAULTS,
        "fusion_dim": 64, "temporal_width": 32, "num_heads": 4,
        "audio_encoder": "dilated_tcn", "vision_encoder": "transformer",
        "pooling": "masked_attention", "fusion": "cross_attention",
        "dropout": 0.1, "load_pretrained": True,
    }
    model = V5Model(model_dir, cfg).to(device)
    vocab_size = int(model.text_encoder.config.vocab_size)
    text_ids = torch.randint(103, vocab_size, (2, 12), device=device)
    text_ids[:, 0] = 101
    text_ids[:, -1] = 102
    text_mask = torch.ones((2, 12), dtype=torch.bool, device=device)
    pool_mask = text_mask.clone()
    pool_mask[:, 0] = False
    pool_mask[:, -1] = False
    audio_mask = torch.ones((2, 50), dtype=torch.bool, device=device)
    vision_mask = torch.ones((2, 50), dtype=torch.bool, device=device)
    audio_mask[0, 18:26] = False
    vision_mask[1, :] = False
    text_mask[1, :] = False
    pool_mask[1, :] = False
    audio = torch.randn((2, 50, 74), device=device)
    vision = torch.randn((2, 50, 35), device=device)
    batch = {
        "text_ids": text_ids, "text_mask": text_mask, "text_pool_mask": pool_mask,
        "text_segments": torch.zeros_like(text_ids), "audio": audio,
        "audio_mask": audio_mask, "vision": vision, "vision_mask": vision_mask,
        "class_label": torch.tensor([2, 1], device=device),
        "regression_label": torch.tensor([1.0, 0.0], device=device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    output_train = model(batch)
    losses = supervised_losses(output_train, batch, neutral_weight=0.25)
    losses["total"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(prefix="v5_cpu_smoke_") as temp:
        checkpoint = Path(temp) / "smoke.safetensors"
        _atomic_safetensors(inference_state_dict(model), checkpoint)
        reload_cfg = {**cfg, "load_pretrained": False}
        reloaded = V5Model(model_dir, reload_cfg, with_distill_heads=False).to(device)
        state = load_file(str(checkpoint), device=str(device))
        missing, unexpected = reloaded.load_state_dict(state, strict=False)
        if [x for x in missing if not x.startswith("distill_adapters.")] or unexpected:
            raise AssertionError(f"Smoke reload mismatch: {missing} {unexpected}")
        del state
        model.eval()
        reloaded.eval()
        with torch.inference_mode():
            expected = model(batch)
            actual = reloaded(batch)
        torch.testing.assert_close(expected["logits"], actual["logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(expected["intensity_raw"], actual["intensity_raw"], rtol=1e-5, atol=1e-6)
        mutated = {key: value.clone() if isinstance(value, torch.Tensor) else value
                   for key, value in batch.items()}
        mutated["text_ids"][~mutated["text_mask"]] = vocab_size + 1000
        mutated["audio"][~mutated["audio_mask"]] = float("nan")
        mutated["vision"][~mutated["vision_mask"]] = float("nan")
        with torch.inference_mode():
            masked_expected = reloaded(batch)
            masked_actual = reloaded(mutated)
        torch.testing.assert_close(masked_expected["logits"], masked_actual["logits"], rtol=0, atol=1e-6)
        torch.testing.assert_close(masked_expected["intensity_raw"], masked_actual["intensity_raw"], rtol=0, atol=1e-6)
        training_cycle = _smoke_training_cycle(model_dir, device, Path(temp))
        gc.collect()
        summary = {
            "status": "passed", "device": str(device), "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "backbone_dir": str(model_dir),
            "model_parameters": model_parameter_count(model),
            "tested_modes": cfg, "forward_backward": True, "save_reload": True,
            "masked_input_invariant": True, "valid_evaluated": False,
            "synthetic_training_cycle": training_cycle,
            "real_train_cli_path_exercised": False,
            "official_test_evaluated": False,
        }
    out_dir = resolve_project_path(output) / "smoke"
    write_json(out_dir / f"{device.type}_smoke.json", summary)
    return summary


def select_best(candidates: list[str | Path], *, output: str | Path) -> dict:
    records = []
    for value in candidates:
        path = Path(value)
        if path.is_dir():
            path = path / "candidate_summary.json"
        row = read_json(path)
        if row.get("official_test_evaluated", False):
            raise ValueError("Candidate summary must be frozen from validation before test")
        records.append({"path": str(path), "candidate": row,
                        "rank": selection_rank(row["complete"])})
    if not records:
        raise ValueError("select requires at least one candidate summary")
    winner = max(records, key=lambda row: row["rank"])
    result = {
        "selected_candidate": winner["candidate"]["candidate"],
        "selected_path": winner["path"],
        "selected_rank": list(winner["rank"]),
        "candidates": [{"path": row["path"], "candidate": row["candidate"].get("candidate"),
                        "rank": list(row["rank"]), "complete": row["candidate"]["complete"]}
                       for row in records],
        "rule": ["complete validation accuracy", "macro F1", "lower MAE"],
        "official_test_evaluated": False,
    }
    write_json(output, result)
    return result


def _synthetic_batch(model, device, seed: int, *, missing: bool, labels=(0, 1)) -> dict:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    vocab_size = int(model.text_encoder.config.vocab_size)
    text_ids = torch.randint(103, vocab_size, (2, 10), generator=generator, device=device)
    text_ids[:, 0], text_ids[:, -1] = 101, 102
    text_mask = torch.ones((2, 10), dtype=torch.bool, device=device)
    text_pool = text_mask.clone()
    text_pool[:, 0] = False
    text_pool[:, -1] = False
    audio_mask = torch.ones((2, 50), dtype=torch.bool, device=device)
    vision_mask = torch.ones((2, 50), dtype=torch.bool, device=device)
    if missing:
        text_mask[:, 3:5] = False
        text_pool[:, 3:5] = False
        text_ids[:, 3:5] = 0
        audio_mask[:, 7:13] = False
        vision_mask[:, 18:22] = False
    return {
        "text_ids": text_ids, "text_mask": text_mask, "text_pool_mask": text_pool,
        "text_segments": torch.zeros_like(text_ids),
        "audio": torch.randn((2, 50, 74), generator=generator, device=device),
        "audio_mask": audio_mask,
        "vision": torch.randn((2, 50, 35), generator=generator, device=device),
        "vision_mask": vision_mask,
        "class_label": torch.as_tensor(labels, dtype=torch.long, device=device),
        "regression_label": torch.as_tensor([-1.0 if labels[0] == 0 else 0.0,
                                               1.0 if labels[1] == 2 else 0.0],
                                              dtype=torch.float32, device=device),
    }


def _smoke_training_cycle(model_dir: Path, device: torch.device, temp: Path) -> dict:
    """Exercise two-batch optimize/save/resume/valid using the shared objective."""
    cfg = {
        **MODEL_DEFAULTS, "fusion_dim": 32, "temporal_width": 16, "num_heads": 4,
        "audio_encoder": "baseline", "vision_encoder": "baseline",
        "pooling": "masked_mean", "fusion": "pooled", "dropout": 0.1,
        "load_pretrained": False,
    }
    seed_everything(1907)
    student = V5Model(model_dir, cfg).to(device)
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=False)
    distill_cfg = {"temperature": 2.0, "feature_components": ["text", "audio", "vision", "fused"],
                   "feature_weight": 0.05, "logit_weight": 0.2}
    training_cfg = {"regression_loss_weight": 0.2, "neutral_aux_weight": 0.25}
    train_pairs = [(_synthetic_batch(student, device, 11, missing=True),
                    _synthetic_batch(student, device, 11, missing=False)),
                   (_synthetic_batch(student, device, 12, missing=True),
                    _synthetic_batch(student, device, 12, missing=False))]
    valid_batches = [_synthetic_batch(student, device, 21, missing=True)]

    def one_epoch(model_to_train, opt, sched, pairs):
        model_to_train.train()
        opt.zero_grad(set_to_none=True)
        sizes = [int(pair[0]["text_ids"].shape[0]) for pair in pairs]
        for index, (student_batch, teacher_batch) in enumerate(pairs):
            with torch.inference_mode():
                teacher_output = teacher(teacher_batch)
            student_output = model_to_train(student_batch)
            losses = training_objective(student_output, student_batch,
                                        teacher_output=teacher_output,
                                        training=training_cfg, distillation=distill_cfg)
            backward_sample_weighted(losses["total"], sizes[index], sizes, None)
        torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        return float(losses["total"].detach().float().item())

    loss1 = one_epoch(student, optimizer, scheduler, train_pairs)
    config_hash = canonical_hash(cfg)
    source_hashes = {"smoke": "synthetic"}
    model_hashes = {"backbone_config": sha256_file(model_dir / "config.json")}
    contract = {"synthetic_splits": {"train": 4, "valid": 2}, "teacher": "frozen-random-copy"}
    state = {
        "model_state": student.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "amp_scaler_state": amp_scaler.state_dict(),
        "completed_epoch": 1, "rng_state": rng_state(), "config_hash": config_hash,
        "source_hashes": source_hashes, "model_hashes": model_hashes,
        "resume_contract": contract, "sampler_state": {"next_epoch": 1},
    }
    checkpoint = temp / "synthetic_epoch1.pt"
    _atomic_torch_save(state, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

    uninterrupted = V5Model(model_dir, cfg).to(device)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-4)
    uninterrupted_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(uninterrupted_optimizer, T_max=2)
    uninterrupted_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restore_training_state(copy.deepcopy(loaded), uninterrupted, uninterrupted_optimizer,
                           uninterrupted_scheduler, uninterrupted_scaler,
                           expected_contract=contract, config_hash=config_hash,
                           source_hashes=source_hashes, model_hashes=model_hashes)
    resumed = V5Model(model_dir, cfg).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=2)
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restore_training_state(copy.deepcopy(loaded), resumed, resumed_optimizer, resumed_scheduler, resumed_scaler,
                           expected_contract=contract, config_hash=config_hash,
                           source_hashes=source_hashes, model_hashes=model_hashes)
    next_rng = copy.deepcopy(loaded["rng_state"])
    restore_rng_state(next_rng)
    loss2_control = one_epoch(uninterrupted, uninterrupted_optimizer, uninterrupted_scheduler, train_pairs)
    restore_rng_state(next_rng)
    loss2_resumed = one_epoch(resumed, resumed_optimizer, resumed_scheduler, train_pairs)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    resumed.eval()
    logits, intensities = [], []
    with torch.inference_mode():
        for batch in valid_batches:
            output = resumed(batch)
            logits.append(output["logits"].float().cpu().numpy())
            intensities.append(output["intensity_raw"].float().cpu().numpy())
    val_logits = np.concatenate(logits)
    val_intensity = np.concatenate(intensities)
    synthetic_metrics = metric_record([0, 1], [-1.0, 0.0], val_logits, val_intensity)
    return {
        "synthetic_batches_per_epoch": 2, "first_epoch_loss": loss1,
        "resume_epoch_loss_control": loss2_control, "resume_epoch_loss_loaded": loss2_resumed,
        "resume_exact_parameter_match": True, "epoch_checkpoint_saved_and_loaded": True,
        "validation_metrics": synthetic_metrics, "official_test_evaluated": False,
    }


def export_package(config_path: str | Path, checkpoint_path: str | Path,
                   scaler_path: str | Path, *, output: str | Path,
                   precision: str = "fp16") -> dict:
    """Delegate complete offline packaging and isolated reload to v5_export."""
    from v5_export import export_package as exporter
    return exporter(config_path, checkpoint_path, scaler_path, output=output, precision=precision)


def freeze_model(config_path: str | Path, validation_path: str | Path,
                 package_path: str | Path, *, output: str | Path) -> dict:
    """Freeze only a package whose independently reloaded validation was measured."""
    validation = read_json(resolve_project_path(validation_path))
    if validation.get("scenario_set") != "v4_32" or validation.get("scenario_count") != 32:
        raise ValueError("Freeze requires all 32 v4-comparable validation scenarios")
    from v5_export import freeze_model as freezer
    return freezer(config_path, validation_path, package_path, output=output)


def load_official_test_once(feature_path: str | Path, freeze_manifest_path: str | Path,
                            official_result_path: str | Path, *, package_claim: str | None = None):
    freeze_path = resolve_project_path(freeze_manifest_path)
    result_path = resolve_project_path(official_result_path)
    if result_path.exists():
        raise FileExistsError("Official test evaluation already exists; it is one-time and cannot be repeated")
    frozen = read_json(freeze_path)
    if frozen.get("status") != "frozen" or frozen.get("official_test_evaluated") is not False:
        raise ValueError("Official test requires a validation-only frozen manifest")
    from v5_export import hash_path
    package = resolve_project_path(frozen["package"])
    package_sha = hash_path(package)
    if package_claim is None:
        raise PermissionError("Official test labels can only be opened by the gated final_test entrypoint")
    marker = freeze_path.parent / "official_test_once" / f"{package_sha}.json"
    if package_claim != package_sha or not marker.is_file():
        raise PermissionError("The package-hash one-shot claim is missing")
    marker_state = read_json(marker)
    if marker_state.get("status") != "running" or marker_state.get("package_sha256") != package_sha:
        raise PermissionError("The package-hash one-shot claim is not active")
    if package_sha != frozen["package_sha256"]:
        raise ValueError("Frozen package hash differs; refusing official test evaluation")
    with Path(feature_path).open("rb") as handle:
        raw = __import__("pickle").load(handle)
    # The gate has passed. Remove train and valid immediately; only now return test.
    raw.pop("train", None)
    raw.pop("valid", None)
    if "test" not in raw:
        raise ValueError("Official test split is missing")
    test_part = raw.pop("test")
    del raw
    return AlignedDataset(AlignedDataset._select(test_part), labeled=True,
                          source="attachment2/test-final-only", special_token_ids=(101, 102))


def final_test(config_path: str | Path, freeze_manifest_path: str | Path,
               package_path: str | Path, *, feature_path: str | Path,
               output: str | Path) -> dict:
    result_path = resolve_project_path(output)
    # Check the one-time output gate before opening the competition pickle.
    if result_path.exists():
        raise FileExistsError("Official test output already exists; refusing a second evaluation")
    freeze_path = resolve_project_path(freeze_manifest_path)
    frozen = read_json(freeze_path)
    package = resolve_project_path(package_path)
    if frozen.get("status") != "frozen" or frozen.get("official_test_evaluated") is not False:
        raise ValueError("Official test requires a frozen package not previously evaluated")
    from v5_export import hash_path, load_offline_predictor
    package_sha = hash_path(package)
    if frozen.get("package_sha256") != package_sha:
        raise ValueError("Package does not match the frozen validation candidate")

    # Preflight model loading and device availability before claiming the
    # irreversible test-read slot. A broken package/device must not consume it.
    predictor = load_offline_predictor(package, device="cuda" if torch.cuda.is_available() else "cpu")
    if result_path.exists():
        raise FileExistsError("Official test output appeared during preflight; refusing to read test")
    # The package-hash marker is now the first irreversible action. The test
    # pickle is opened only after this exclusive claim succeeds.
    marker = freeze_path.parent / "official_test_once" / f"{package_sha}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError("Official test was already claimed for this frozen package hash") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"status": "running", "package_sha256": package_sha,
                   "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
        handle.write("\n")
    test_ds = load_official_test_once(feature_path, freeze_manifest_path, result_path,
                                      package_claim=package_sha)
    samples = [test_ds[i] for i in range(len(test_ds))]
    predictions = predictor.predict_samples(samples)
    y = test_ds.part["classification_labels"].astype(int)
    r = test_ds.part["regression_labels"].astype(float)
    metrics = metric_record(y, r, predictions["logits"], predictions["intensity_raw"])
    result = {
        "status": "evaluated_once_after_freeze", "package_sha256": package_sha,
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "official_test": metrics, "official_test_evaluated": True,
        "no_model_selection_after_test": True,
    }
    write_json(result_path, result)
    frozen["official_test_evaluated"] = True
    frozen["official_test_result_sha256"] = sha256_file(result_path)
    frozen["official_test_completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    freeze_temp = freeze_path.with_suffix(freeze_path.suffix + ".tmp")
    write_json(freeze_temp, frozen)
    os.replace(freeze_temp, freeze_path)
    write_json(marker, {"status": "complete", "package_sha256": package_sha,
                        "result_sha256": sha256_file(result_path),
                        "completed_at_utc": frozen["official_test_completed_at_utc"]})
    return result


def evaluate_package(package_path: str | Path, *, feature_path: str | Path | None = None,
                     output: str | Path, scenario_set: str = "v4_32") -> dict:
    """Score the exact exported offline package on validation only."""
    from v5_export import hash_path, load_offline_predictor, verify_package
    package = resolve_project_path(package_path)
    verification = verify_package(package, max_bytes=PACKAGE_MAX_BYTES)
    if verification.get("status") != "passed":
        raise ValueError("Exported package verification did not pass")
    train_ds, valid_ds = load_train_valid(feature_path)
    del train_ds
    samples = [valid_ds[i] for i in range(len(valid_ds))]
    y = valid_ds.part["classification_labels"].astype(int)
    r = valid_ds.part["regression_labels"].astype(float)
    predictor = load_offline_predictor(package, device="cuda" if torch.cuda.is_available() else "cpu")
    if scenario_set not in {"v4_32", "quick_all_random"}:
        raise ValueError("scenario_set must be v4_32 or quick_all_random")
    definitions = validation_case_definitions(101)
    if scenario_set == "quick_all_random":
        definitions = [definitions[0]] + [case for case in definitions
                                          if case["name"] in {"all|random|0.1", "all|random|0.5", "all|random|0.7"}]
    metrics_by_case = {}
    for case in definitions:
        current_samples = validation_case_samples(samples, case)
        output_values = predictor.predict_samples(current_samples)
        metrics_by_case[case["name"]] = metric_record(
            y, r, output_values["logits"], output_values["intensity_raw"])
    complete = metrics_by_case.pop("complete")
    missing_mean = float(np.mean([row["accuracy"] for row in metrics_by_case.values()]))
    quick_all_random = {name: metrics_by_case[name] for name in
                        ("all|random|0.1", "all|random|0.5", "all|random|0.7")
                        if name in metrics_by_case}
    result = {
        "complete": complete, "missing_all_random": quick_all_random,
        "missing_scenarios": metrics_by_case,
        "missing_scenario_count": len(metrics_by_case),
        "missing_scenario_mean_accuracy": missing_mean,
        "scenario_set": scenario_set,
        "scenario_count": len(metrics_by_case) + 1,
        "selection_order": ["complete_accuracy", "macro_f1", "lower_mae"],
        "package": str(package), "package_sha256": hash_path(package),
        "package_reload_verified": bool(verification.get("package_reload_verified", False)),
        "package_verification": verification,
        "fixed_valid_count": len(valid_ds), "official_test_evaluated": False,
    }
    if not result["package_reload_verified"]:
        raise ValueError("Package has not passed its independent offline reload check")
    write_json(output, result)
    return result


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke", help="synthetic forward/backward/save/reload test; no dataset access")
    p.add_argument("--model-dir", default="models/bert_tiny")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p = sub.add_parser("prepare", help="fit train-only AV scaler and record train/valid fingerprints")
    p.add_argument("--config", default="configs/v5/stage00_baseline_bert_mini.json")
    p.add_argument("--feature-path")
    p.add_argument("--output")
    p = sub.add_parser("train", help="GPU-only train one candidate; no official test access")
    p.add_argument("--config", default="configs/v5/stage00_baseline_bert_mini.json")
    p.add_argument("--feature-path")
    p.add_argument("--output")
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--teacher-config")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate", help="evaluate one checkpoint on official valid only")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--feature-path")
    p.add_argument("--output")
    p.add_argument("--scenario-set", choices=("v4_32", "quick_all_random"), default="v4_32")
    p = sub.add_parser("evaluate-package", help="evaluate the exact exported/offline-reloaded package on valid only")
    p.add_argument("--package", required=True)
    p.add_argument("--feature-path")
    p.add_argument("--output", required=True)
    p.add_argument("--scenario-set", choices=("v4_32", "quick_all_random"), default="v4_32")
    p = sub.add_parser("select", help="rank candidate summaries by valid Accuracy, Macro-F1, then MAE")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT / "selection.json"))
    p = sub.add_parser("export", help="export and measure complete offline single-student package")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=("fp16",), default="fp16")
    p = sub.add_parser("freeze", help="freeze one validation-selected package before test")
    p.add_argument("--config", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("final-test", help="one-time official test evaluation after freeze")
    p.add_argument("--config", required=True)
    p.add_argument("--freeze-manifest", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--feature-path", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("predict-attachment3", help="predict unlabeled aligned attachment 3 offline")
    p.add_argument("--package", required=True)
    p.add_argument("--folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    args = _cli().parse_args(argv)
    if args.command == "smoke":
        result = run_smoke(args.model_dir, output=args.output, device_name=args.device)
    elif args.command == "prepare":
        result = {"output": str(prepare(args.config, output=args.output, feature_path=args.feature_path)),
                  "official_test_evaluated": False}
    elif args.command == "train":
        result = train(args.config, output=args.output, feature_path=args.feature_path,
                       teacher_checkpoint=args.teacher_checkpoint,
                       teacher_config_path=args.teacher_config, resume=args.resume)
        if not isinstance(result, dict) or not result.get("checkpoint_sha256"):
            raise RuntimeError("Training returned no checkpoint summary; refusing a successful null result")
    elif args.command == "evaluate":
        result = evaluate_validation(args.config, args.checkpoint,
                                     output=args.output, feature_path=args.feature_path,
                                     scenario_set=args.scenario_set)
    elif args.command == "evaluate-package":
        result = evaluate_package(args.package, feature_path=args.feature_path,
                                  output=args.output, scenario_set=args.scenario_set)
    elif args.command == "select":
        result = select_best(args.candidate, output=args.output)
    elif args.command == "export":
        result = export_package(args.config, args.checkpoint, args.scaler,
                                output=args.output, precision=args.precision)
    elif args.command == "freeze":
        result = freeze_model(args.config, args.validation, args.package, output=args.output)
    elif args.command == "final-test":
        result = final_test(args.config, args.freeze_manifest, args.package,
                            feature_path=args.feature_path, output=args.output)
    elif args.command == "predict-attachment3":
        from v5_export import predict_attachment3
        result = predict_attachment3(args.package, args.folder, args.output, device=args.device)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
