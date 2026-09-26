"""Build and verify a self-contained, offline v5 single-student package.

The exporter stores one FP16 copy of the complete inference model. The model
is reconstructed from the packaged Hugging Face config with
``load_pretrained=false`` and then loaded from the bundled state dict. No model
weights are fetched from the original backbone directory or from the network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from aligned_dataset import AlignedDataset
from pipeline import MaskedStandardizer
from v5_data import collate_samples, load_tokenizers
from v5_model import MODEL_DEFAULTS, V5Model, inference_state_dict, model_parameter_count


ROOT = Path(__file__).resolve().parent
EXPORT_VERSION = "codex-v5-offline-student-v1"
PACKAGE_MAX_BYTES = 50_000_000
CLASS_ORDER = ("Negative", "Neutral", "Positive")
RUNTIME_FILES = (
    "v5_model.py",
    "v5_data.py",
    "aligned_dataset.py",
    "p2.py",
    "pipeline.py",
)
SENSITIVE_KEYS = {
    "_name_or_path", "name_or_path", "token", "auth_token", "access_token",
    "use_auth_token", "hf_token", "api_key", "password", "secret",
    "credentials", "authorization", "cache_dir",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json_read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_write(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: str | Path) -> str:
    """Hash a file directly or a directory as sorted relative-path/hash pairs."""
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    # Sort by the canonical POSIX relative path, not Path's platform-specific
    # ordering (WindowsPath is case-insensitive while PosixPath is not).
    for file_path in sorted(
        (p for p in target.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(target).as_posix(),
    ):
        rel = file_path.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def _safe_relpath(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Unsafe package path: {value!r}")
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Drive-qualified package path is not allowed: {value!r}")
    return rel


def _safe_extract(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            _safe_relpath(member.filename)
        archive.extractall(target)


def _file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "text/x-python",
        ".json": "application/json",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".safetensors": "application/x-safetensors; tensors=float16",
        ".npz": "application/x-npz",
    }.get(suffix, "application/octet-stream")


def _sensitive_scrub(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        clean = {}
        for child_key, child in value.items():
            lowered = str(child_key).lower()
            if lowered in SENSITIVE_KEYS:
                continue
            cleaned = _sensitive_scrub(child, str(child_key))
            if cleaned is not _DROP:
                clean[child_key] = cleaned
        return clean
    if isinstance(value, list):
        items = [_sensitive_scrub(item, key) for item in value]
        return [item for item in items if item is not _DROP]
    if isinstance(value, str):
        # Eliminate accidental machine paths in tokenizer/config metadata.
        if re.search(r"(?:^|\s)[A-Za-z]:[\\/]|(?:^|\s)\\\\[^\\]+\\", value):
            return _DROP
        return value
    return value


class _Drop:
    pass


_DROP = _Drop()


def _model_config(config: dict) -> dict:
    result = dict(MODEL_DEFAULTS)
    result.update(config.get("model", {}))
    if "backbone_dir" not in result and "backbone_dir" in config:
        result["backbone_dir"] = config["backbone_dir"]
    if not result.get("backbone_dir"):
        raise ValueError("Training config needs model.backbone_dir for local export")
    return result


def _load_model_for_export(config: dict, checkpoint_path: Path, backbone: Path) -> V5Model:
    model_cfg = _model_config(config)
    # The source config initializes architecture only. Every trained parameter,
    # including the backbone, comes from the single checkpoint file.
    export_cfg = dict(model_cfg)
    export_cfg.update({"load_pretrained": False, "backbone_dir": str(backbone),
                       "teacher_feature_dim": None})
    model = V5Model(backbone, export_cfg, with_distill_heads=False).cpu()
    state = load_file(str(checkpoint_path), device="cpu")
    state = {str(name).removeprefix("module."): value for name, value in state.items()
             if not str(name).startswith("distill_adapters.")
             and not str(name).startswith("module.distill_adapters.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.startswith("distill_adapters.")]
    if missing or unexpected:
        raise ValueError(f"Checkpoint/model mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    model.eval()
    return model


def _copy_source_assets(config: dict, model_cfg: dict, build_dir: Path,
                        backbone_source: Path, scaler_path: Path) -> tuple[Path, Path]:
    backbone_dir = build_dir / "backbone"
    tokenizer_dir = build_dir / "tokenizer"
    source_vocab_dir = build_dir / "source_vocab"
    backbone_dir.mkdir()
    tokenizer_dir.mkdir()
    source_vocab_dir.mkdir()

    source_hf_config = AutoConfig.from_pretrained(str(backbone_source), local_files_only=True)
    config_dict = _sensitive_scrub(source_hf_config.to_dict())
    config_dict.pop("_name_or_path", None)
    config_dict.pop("name_or_path", None)
    _json_write(backbone_dir / "config.json", config_dict)

    # Save tokenizer assets alone; AutoTokenizer does not serialize backbone
    # weights. The tokenizer's JSON asset is model data: its ``model.vocab``
    # keys are ordinary token strings, some of which happen to look like
    # credential field names (for example "token" and "secret"). Never run
    # the generic metadata scrubber over tokenizer.json or vocab/model assets.
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_source), local_files_only=True, use_fast=True)
    tokenizer.save_pretrained(tokenizer_dir)
    for metadata in tokenizer_dir.glob("*.json"):
        if metadata.name == "tokenizer.json":
            # Parse to reject malformed output, then preserve every serialized
            # vocabulary item and id byte-for-byte.
            try:
                json.loads(metadata.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError(f"Tokenizer metadata is not valid JSON: {metadata.name}")
            continue
        try:
            parsed = json.loads(metadata.read_text(encoding="utf-8"))
            _json_write(metadata, _sensitive_scrub(parsed))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError(f"Tokenizer metadata is not valid JSON: {metadata.name}")

    # Fail at export time if serialization or metadata handling changed the
    # target tokenizer's vocabulary. A few lost IDs can silently retokenize a
    # visible input and make an otherwise correct model look badly quantized.
    packaged_tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, use_fast=True)
    source_vocab_map = tokenizer.get_vocab()
    packaged_vocab_map = packaged_tokenizer.get_vocab()
    if source_vocab_map != packaged_vocab_map:
        missing = sorted(set(source_vocab_map) - set(packaged_vocab_map))[:8]
        changed = sorted(token for token in source_vocab_map.keys() & packaged_vocab_map.keys()
                         if source_vocab_map[token] != packaged_vocab_map[token])[:8]
        extra = sorted(set(packaged_vocab_map) - set(source_vocab_map))[:8]
        raise ValueError(
            "Packaged tokenizer vocabulary changed during serialization: "
            f"source={len(source_vocab_map)} packaged={len(packaged_vocab_map)} "
            f"missing={missing} changed_ids={changed} extra={extra}")

    configured_vocab = (model_cfg.get("source_vocab") or config.get("source_vocab")
                        or config.get("data", {}).get("source_vocab"))
    if configured_vocab:
        source_vocab = _resolve(configured_vocab)
    else:
        source_vocab = ROOT / "models" / "bert_mini" / "vocab.txt"
    if not source_vocab.is_file():
        # Some compact local BERT bundles use the same vocabulary as the target.
        candidate = backbone_source / "vocab.txt"
        if not candidate.is_file():
            raise FileNotFoundError(f"Source BERT vocabulary is required: {source_vocab}")
        source_vocab = candidate
    shutil.copyfile(source_vocab, source_vocab_dir / "vocab.txt")
    shutil.copyfile(scaler_path, build_dir / "standardizer.json")
    # Parse and validate at export time so malformed scalers do not create a
    # package that only fails when the user first runs inference.
    MaskedStandardizer.load(build_dir / "standardizer.json")
    return tokenizer_dir, source_vocab_dir / "vocab.txt", {
        "source": "BertTokenizer(source_vocab/vocab.txt; do_lower_case=true)",
        "target": "AutoTokenizer.from_pretrained(tokenizer; local_files_only=true; use_fast=true)",
        "target_class": packaged_tokenizer.__class__.__name__,
        "target_is_fast": bool(getattr(packaged_tokenizer, "is_fast", False)),
    }


def _inference_script() -> str:
    return r'''"""Offline inference entry point generated inside each v5 package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import BertTokenizer, AutoTokenizer

from aligned_dataset import AlignedDataset
from pipeline import MaskedStandardizer
from v5_data import collate_samples
from v5_model import V5Model


ROOT = Path(__file__).resolve().parent


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "codex-v5-offline-student-v1" or manifest.get("status") != "qualified":
        raise ValueError("Package manifest is absent or not qualified")
    expected = manifest.get("files", {})
    for rel in expected:
        safe = PurePosixPath(rel)
        if safe.is_absolute() or ".." in safe.parts or not safe.parts:
            raise ValueError(f"Unsafe manifest path: {rel!r}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(expected):
        extras = sorted(actual - set(expected))[:8]
        missing = sorted(set(expected) - actual)[:8]
        raise ValueError(f"Package file set differs from manifest: extra={extras}, missing={missing}")
    for rel, info in expected.items():
        path = root / PurePosixPath(rel)
        if _sha256(path) != info.get("sha256") or path.stat().st_size != info.get("bytes"):
            raise ValueError(f"Package file integrity check failed: {rel}")
    config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    if config.get("load_pretrained") is not False or config.get("backbone_dir") != "backbone":
        raise ValueError("Package must initialize from its local config with load_pretrained=false")
    return manifest, config


class OfflinePredictor:
    def __init__(self, package_dir, device="cpu"):
        self.root = Path(package_dir).resolve()
        self.manifest, self.model_cfg = _manifest(self.root)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.tokenizer_source = BertTokenizer(
            vocab_file=str(self.root / "source_vocab" / "vocab.txt"), do_lower_case=True)
        self.tokenizer_target = AutoTokenizer.from_pretrained(
            str(self.root / "tokenizer"), local_files_only=True, use_fast=True)
        self.standardizer = MaskedStandardizer.load(self.root / "standardizer.json")
        cfg = dict(self.model_cfg)
        cfg.update({"backbone_dir": str(self.root / "backbone"), "load_pretrained": False,
                    "teacher_feature_dim": None})
        self.model = V5Model(self.root / "backbone", cfg, with_distill_heads=False).cpu()
        state = load_file(str(self.root / "student.safetensors"), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [name for name in missing if not name.startswith("distill_adapters.")]
        if missing or unexpected:
            raise ValueError(f"Packaged state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        self.model.to(self.device).eval()
        self.max_text_length = int(self.manifest["inference"]["max_text_length"])
        self.feature_clip = float(self.manifest["inference"]["feature_clip"])

    def predict_samples(self, samples, batch_size=32):
        if not samples:
            return {"logits": np.zeros((0, 3), dtype=np.float32),
                    "intensity_raw": np.zeros(0, dtype=np.float32)}
        all_logits, all_intensity = [], []
        with torch.inference_mode():
            for start in range(0, len(samples), int(batch_size)):
                batch = collate_samples(
                    samples[start:start + int(batch_size)], self.standardizer,
                    self.tokenizer_source, self.tokenizer_target, self.max_text_length,
                    device=self.device, feature_clip=self.feature_clip)
                output = self.model(batch)
                all_logits.append(output["logits"].float().cpu().numpy())
                all_intensity.append(output["intensity_raw"].float().cpu().numpy())
        return {"logits": np.concatenate(all_logits),
                "intensity_raw": np.concatenate(all_intensity)}

    def predict_arrays(self, arrays, batch_size=32):
        if any("label" in str(key).lower() for key in arrays):
            raise ValueError("Offline prediction input must not contain dataset labels")
        required = {"text_bert", "audio", "vision"}
        if not required.issubset(arrays):
            raise ValueError(f"Input NPZ needs keys {sorted(required)}")
        part = {key: arrays[key] for key in required}
        if "id" in arrays:
            part["id"] = arrays["id"]
        specials = tuple(int(x) for x in (
            self.tokenizer_source.pad_token_id, self.tokenizer_source.cls_token_id,
            self.tokenizer_source.sep_token_id))
        dataset = AlignedDataset(part, labeled=False, source="offline-inference",
                                 special_token_ids=specials)
        samples = [dataset[i] for i in range(len(dataset))]
        result = self.predict_samples(samples, batch_size=batch_size)
        result["sample_ids"] = np.asarray([sample["sample_id"] for sample in samples], dtype=str)
        return result


def load_offline_predictor(package_path, device="cpu"):
    path = Path(package_path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.namelist():
                rel = PurePosixPath(member)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe path in package ZIP")
        # Keep extracted files alive for as long as the predictor exists.
        import tempfile
        temp = tempfile.TemporaryDirectory(prefix="v5_offline_")
        with zipfile.ZipFile(path, "r") as archive:
            archive.extractall(temp.name)
        predictor = OfflinePredictor(temp.name, device=device)
        predictor._temporary_directory = temp
        return predictor
    return OfflinePredictor(path, device=device)


def _save_predictions(predictions, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = predictions["sample_ids"]
    logits = predictions["logits"]
    values = predictions["intensity_raw"]
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, sample_ids=ids, logits=logits, intensity_raw=values)
        return
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "negative_probability", "neutral_probability",
                         "positive_probability", "predicted_class", "intensity_raw", "intensity_clipped"])
        for sample_id, row, intensity in zip(ids, probs, values):
            writer.writerow([sample_id, *[float(x) for x in row], int(row.argmax()),
                             float(intensity), float(np.clip(intensity, -3.0, 3.0))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with text_bert, audio, vision, optional id")
    parser.add_argument("--output", required=True, help="CSV or NPZ predictions path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    with np.load(args.input, allow_pickle=False) as data:
        predictions = load_offline_predictor(ROOT, args.device).predict_arrays(data, args.batch_size)
    _save_predictions(predictions, args.output)
    print(json.dumps({"rows": len(predictions["sample_ids"]), "output": str(Path(args.output).resolve()),
                      "official_test_evaluated": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _copy_runtime(build_dir: Path) -> None:
    for name in RUNTIME_FILES:
        src = ROOT / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing inference runtime source: {src}")
        shutil.copyfile(src, build_dir / name)
    (build_dir / "infer.py").write_text(_inference_script(), encoding="utf-8")
    (build_dir / "requirements-inference.txt").write_text(
        "numpy>=1.23\nscikit-learn>=1.2\ntorch>=2.2\ntransformers>=4.46,<5\n"
        "tokenizers>=0.20\nsafetensors>=0.4\n", encoding="utf-8")
    (build_dir / "README.md").write_text(
        "# Offline v5 student package\n\n"
        "This package contains one student model only. `student.safetensors` stores all inference "
        "weights, including the text backbone, once in FP16. The local Hugging Face config is used "
        "to construct the architecture with `load_pretrained=false`; inference then loads the bundled "
        "state dict. The original backbone directory and network are not needed.\n\n"
        "## Install and run\n\n"
        "Install CPU-compatible dependencies with `python -m pip install -r requirements-inference.txt`. "
        "From this folder, run `python infer.py --input features.npz --output predictions.csv --device cpu`. "
        "The NPZ must contain `text_bert` shaped `(N,3,50)`, `audio` shaped `(N,50,74)`, and `vision` "
        "shaped `(N,50,35)`; `id` is optional. Do not include labels. The project CLI also offers an "
        "attachment 3 prediction command.\n\n"
        "The serialized parameters are FP16; the loader expands them into the model's standard runtime "
        "dtype, so CPU inference uses ordinary FP32 operators. INT8 is unsupported. Model outputs are "
        "three class logits and raw intensity; the CSV also includes softmax probabilities and intensity "
        "clipped to [-3,3]. Audio and vision are normalized by the included train-fitted standardizer, "
        "masked positions stay zero, and feature values are clipped using the recorded inference setting.\n\n"
        "`manifest.json` records SHA256 and byte count for every package file except itself (self-hashing "
        "would be recursive). The directory tree hash covers the manifest too. No teacher weights, optimizer "
        "state, training checkpoint, source data, or labels are included.\n", encoding="utf-8")


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in RUNTIME_FILES}


def _make_manifest(build_dir: Path, model: V5Model, model_cfg: dict,
                   config: dict, scaler_path: Path, checkpoint_path: Path,
                   tokenizer_strategy: dict,
                   precision: str) -> dict:
    files = {}
    for path in sorted(p for p in build_dir.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(build_dir).as_posix()
        files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "format": "application/x-safetensors; tensors=float16" if rel == "student.safetensors" else _file_format(path),
            "parameter_count": int(model_parameter_count(model)) if rel == "student.safetensors" else None,
            "version": EXPORT_VERSION,
        }
    data_cfg = config.get("data", {})
    return {
        "format": EXPORT_VERSION,
        "status": "qualified",
        "precision": precision,
        "class_order": list(CLASS_ORDER),
        "model_parameters": int(model_parameter_count(model)),
        "inference": {
            "max_text_length": int(data_cfg.get("max_text_length", 128)),
            "feature_clip": float(data_cfg.get("feature_clip", 5.0)),
            "text_bert_positions": 50,
            "audio_shape": [50, 74],
            "vision_shape": [50, 35],
        },
        "tokenizer_strategy": tokenizer_strategy,
        "files": files,
        # Keep the exporter provenance separate from the trained checkpoint's
        # source hashes: export fixes do not imply that the student was trained
        # again with a different model/training implementation.
        "exporter_sha256": sha256_file(Path(__file__)),
        "source_code_sha256": _source_hashes(),
        "official_test_evaluated": False,
        "training_labels_included": False,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "standardizer_sha256": sha256_file(scaler_path),
        "package_reload_verification": {"verified": False},
    }


def _write_zip(package_dir: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=False) as archive:
        for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
            arcname = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _verify_directory(package_dir: Path, max_bytes: int) -> dict:
    manifest_path = package_dir / "manifest.json"
    if not package_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Not a completed offline student directory: {package_dir}")
    manifest = _json_read(manifest_path)
    if manifest.get("format") != EXPORT_VERSION or manifest.get("status") != "qualified":
        raise ValueError("Package manifest does not mark this format as qualified")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError("Package manifest files table is invalid")
    actual_files = {p.relative_to(package_dir).as_posix(): p for p in package_dir.rglob("*") if p.is_file()}
    expected_names = set(expected)
    if "manifest.json" in expected_names or set(actual_files) != expected_names | {"manifest.json"}:
        raise ValueError("Package directory contents do not match manifest")
    for rel, record in expected.items():
        _safe_relpath(rel)
        path = actual_files[rel]
        if int(record.get("bytes", -1)) != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"SHA256/byte validation failed for {rel}")
        if not record.get("format") or "parameter_count" not in record or not record.get("version"):
            raise ValueError(f"Manifest metadata is incomplete for {rel}")
    total = sum(path.stat().st_size for path in actual_files.values())
    if total >= int(max_bytes):
        raise ValueError(f"Package files total {total:,} bytes; limit is strictly below {max_bytes:,}")
    config = _json_read(package_dir / "model_config.json")
    if config.get("load_pretrained") is not False or config.get("backbone_dir") != "backbone":
        raise ValueError("Package config must use local config and load_pretrained=false")
    if not (package_dir / "backbone" / "config.json").is_file():
        raise ValueError("Local backbone config is missing")
    if any(path.name.lower() in {"pytorch_model.bin", "model.safetensors", "optimizer.pt"}
           for path in package_dir.rglob("*") if path.is_file() and path.name != "student.safetensors"):
        raise ValueError("Package contains duplicate backbone or training weights")
    return {"status": "passed", "files": len(actual_files), "payload_files": len(expected),
            "directory_bytes": total, "directory_sha256": hash_path(package_dir),
            "manifest_sha256": sha256_file(manifest_path), "precision": manifest.get("precision"),
            "package_reload_verified": bool(manifest.get("package_reload_verification", {}).get("verified")),
            "official_test_evaluated": False}


def _verify_zip(zip_path: Path, max_bytes: int, expected_dir: Path | None = None) -> dict:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    zip_bytes = zip_path.stat().st_size
    if zip_bytes >= int(max_bytes):
        raise ValueError(f"ZIP is {zip_bytes:,} bytes; limit is strictly below {max_bytes:,}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        names = [_safe_relpath(item.filename).as_posix() for item in infos]
        if len(names) != len(set(names)) or any(item.is_dir() for item in infos):
            raise ValueError("ZIP has duplicate names or unexpected directory entries")
        uncompressed = sum(item.file_size for item in infos)
        if uncompressed >= int(max_bytes):
            raise ValueError(f"ZIP contents total {uncompressed:,} bytes; limit is strictly below {max_bytes:,}")
        if expected_dir is not None:
            actual = {p.relative_to(expected_dir).as_posix() for p in expected_dir.rglob("*") if p.is_file()}
            if set(names) != actual:
                raise ValueError("ZIP file set differs from exported package directory")
            for name in names:
                with archive.open(name) as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                if digest != sha256_file(expected_dir / PurePosixPath(name)):
                    raise ValueError(f"ZIP content mismatch for {name}")
    return {"zip_bytes": zip_bytes, "zip_sha256": sha256_file(zip_path),
            "zip_uncompressed_bytes": uncompressed}


def verify_package(package_path: str | Path, max_bytes: int = PACKAGE_MAX_BYTES) -> dict:
    """Verify directory or ZIP hashes, file sets, and strict size ceilings."""
    path = _resolve(package_path)
    if path.is_dir():
        result = _verify_directory(path, max_bytes)
        sibling_zip = Path(str(path) + ".zip")
        if sibling_zip.is_file():
            result.update(_verify_zip(sibling_zip, max_bytes, path))
            result["zip_path"] = str(sibling_zip.resolve())
        if result["directory_bytes"] >= max_bytes:
            raise ValueError("Directory byte gate failed")
        return result
    if path.is_file() and path.suffix.lower() == ".zip":
        zip_info = _verify_zip(path, max_bytes)
        with tempfile.TemporaryDirectory(prefix="v5_verify_") as temp:
            _safe_extract(path, Path(temp))
            result = _verify_directory(Path(temp), max_bytes)
        result.update(zip_info)
        result["zip_path"] = str(path.resolve())
        return result
    raise FileNotFoundError(f"Expected package directory or ZIP: {path}")


def _synthetic_arrays(vocab_size: int, count: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260924)
    text = np.zeros((count, 3, 50), dtype=np.int64)
    audio = np.zeros((count, 50, 74), dtype=np.float32)
    vision = np.zeros((count, 50, 35), dtype=np.float32)
    for index in range(count):
        length = 12 + index * 3
        text[index, 0, 0] = 101
        text[index, 0, 1:length - 1] = rng.integers(103, max(104, vocab_size), size=length - 2)
        text[index, 0, length - 1] = 102
        text[index, 1, :length] = 1
        audio[index, :length] = rng.normal(0, 0.25, size=(length, 74)).astype(np.float32)
        vision[index, :length] = rng.normal(0, 0.25, size=(length, 35)).astype(np.float32)
        if index % 2 == 1:
            audio[index, length // 2:length // 2 + 2] = 0
    return {"text_bert": text, "audio": audio, "vision": vision,
            "id": np.asarray([f"synthetic-{i}" for i in range(count)])}


def _source_prediction(model: V5Model, config: dict, scaler_path: Path,
                       backbone: Path, source_vocab: Path,
                       arrays: dict[str, np.ndarray]) -> dict:
    source_tokenizer, target_tokenizer = load_tokenizers(backbone, source_vocab=source_vocab)
    special = tuple(int(x) for x in (source_tokenizer.pad_token_id,
                                     source_tokenizer.cls_token_id, source_tokenizer.sep_token_id))
    dataset = AlignedDataset({key: arrays[key] for key in ("text_bert", "audio", "vision", "id")},
                             labeled=False, source="export-synthetic-validation",
                             special_token_ids=special)
    samples = [dataset[i] for i in range(len(dataset))]
    scaler = MaskedStandardizer.load(scaler_path)
    cfg = config.get("data", {})
    batch = collate_samples(samples, scaler, source_tokenizer, target_tokenizer,
                            int(cfg.get("max_text_length", 128)), device="cpu",
                            feature_clip=float(cfg.get("feature_clip", 5.0)))
    model.eval()
    with torch.inference_mode():
        output = model(batch)
    return {"logits": output["logits"].float().cpu().numpy(),
            "intensity_raw": output["intensity_raw"].float().cpu().numpy()}


def _run_isolated_inference(package_dir: Path, arrays: dict[str, np.ndarray],
                            work_dir: Path) -> dict:
    input_path = work_dir / "synthetic_input.npz"
    output_path = work_dir / "synthetic_predictions.npz"
    np.savez_compressed(input_path, **arrays)
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TRANSFORMERS_CACHE", "HF_HOME"):
        env.pop(key, None)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})
    proc = subprocess.run(
        [sys.executable, str(package_dir / "infer.py"), "--input", str(input_path),
         "--output", str(output_path), "--device", "cpu", "--batch-size", "2"],
        cwd=str(work_dir), env=env, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or not output_path.is_file():
        raise RuntimeError("Independent offline package inference failed: " + proc.stderr[-2000:])
    with np.load(output_path, allow_pickle=False) as result:
        return {key: result[key].copy() for key in result.files}


def _write_model_config(build_dir: Path, config: dict, model_cfg: dict) -> dict:
    exported = dict(model_cfg)
    exported.update({"load_pretrained": False, "backbone_dir": "backbone",
                     "teacher_feature_dim": None, "source_vocab": "source_vocab/vocab.txt"})
    exported = _sensitive_scrub(exported)
    _json_write(build_dir / "model_config.json", exported)
    return exported


def _build_package(config_path: Path, checkpoint_path: Path, scaler_path: Path,
                   build_dir: Path, precision: str, *, max_bytes: int = PACKAGE_MAX_BYTES,
                   verify_process: bool = True) -> tuple[dict, dict]:
    if precision.lower() == "int8":
        raise NotImplementedError("INT8 export is unsupported; use --precision fp16")
    if precision.lower() != "fp16":
        raise ValueError("precision must be fp16")
    config = _json_read(config_path)
    model_cfg = _model_config(config)
    backbone_source = _resolve(model_cfg["backbone_dir"])
    if not backbone_source.is_dir():
        raise FileNotFoundError(f"Local backbone configuration directory is missing: {backbone_source}")
    if not checkpoint_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError("Checkpoint and train-fitted standardizer files are required")
    model = _load_model_for_export(config, checkpoint_path, backbone_source)
    build_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_dir, source_vocab_path, tokenizer_strategy = _copy_source_assets(
        config, model_cfg, build_dir, backbone_source, scaler_path)
    _copy_runtime(build_dir)
    exported_cfg = _write_model_config(build_dir, config, model_cfg)
    del tokenizer_dir, exported_cfg

    state = inference_state_dict(model)
    # Inference parameters are written once, with every floating tensor stored
    # as FP16. CPU reload casts the tensors into ordinary FP32 module storage.
    fp16_state = {key: value.half() if value.is_floating_point() else value
                  for key, value in state.items()}
    save_file(fp16_state, str(build_dir / "student.safetensors"))
    del state, fp16_state

    manifest = _make_manifest(build_dir, model, model_cfg, config, scaler_path,
                              checkpoint_path, tokenizer_strategy, "fp16")
    manifest.pop("_checkpoint_path", None)
    _json_write(build_dir / "manifest.json", manifest)
    before = _verify_directory(build_dir, max_bytes)

    reload_record = {"verified": False}
    if verify_process:
        vocab_size = int(model.text_encoder.config.vocab_size)
        arrays = _synthetic_arrays(vocab_size)
        reference = _source_prediction(model, config, scaler_path, backbone_source,
                                       source_vocab_path, arrays)
        with tempfile.TemporaryDirectory(prefix="v5_export_check_") as temp:
            observed = _run_isolated_inference(build_dir, arrays, Path(temp))
        logits_error = np.abs(reference["logits"] - observed["logits"])
        intensity_error = np.abs(reference["intensity_raw"] - observed["intensity_raw"])
        max_abs = float(max(logits_error.max(initial=0.0), intensity_error.max(initial=0.0)))
        mean_abs = float((logits_error.mean() + intensity_error.mean()) / 2.0)
        tolerance = {"max_abs": 0.05, "mean_abs": 0.01}
        if max_abs > tolerance["max_abs"] or mean_abs > tolerance["mean_abs"]:
            raise ValueError(f"FP16 reload differs from source precision beyond tolerance: max={max_abs:.6g}, mean={mean_abs:.6g}")
        reload_record = {"verified": True, "method": "separate offline infer.py process",
                         "sample_count": int(len(arrays["id"])), "max_abs_difference": max_abs,
                         "mean_abs_difference": mean_abs, "tolerance": tolerance,
                         "official_test_evaluated": False}
    manifest["package_reload_verification"] = reload_record
    _json_write(build_dir / "manifest.json", manifest)
    checked = _verify_directory(build_dir, max_bytes)
    if verify_process:
        # Reload once more after the final manifest is written so the verified
        # artifact is byte-for-byte the package that will be published.
        with tempfile.TemporaryDirectory(prefix="v5_export_final_check_") as temp:
            observed = _run_isolated_inference(build_dir, arrays, Path(temp))
        final_error = float(max(np.abs(reference["logits"] - observed["logits"]).max(initial=0.0),
                                np.abs(reference["intensity_raw"] - observed["intensity_raw"]).max(initial=0.0)))
        if final_error > tolerance["max_abs"]:
            raise ValueError(f"Final package reload failed after manifest finalization: max_abs={final_error:.6g}")
        reload_record["final_manifest_reload_max_abs_difference"] = final_error
        manifest["package_reload_verification"] = reload_record
        _json_write(build_dir / "manifest.json", manifest)
        checked = _verify_directory(build_dir, max_bytes)
    return manifest, checked


def export_package(config_path: str | Path, checkpoint_path: str | Path,
                   scaler_path: str | Path, output: str | Path,
                   precision: str = "fp16", *, max_bytes: int = PACKAGE_MAX_BYTES,
                   verify_process: bool = True) -> dict:
    """Atomically export a package directory and sibling ZIP after all gates pass."""
    config_path = _resolve(config_path)
    checkpoint_path = _resolve(checkpoint_path)
    scaler_path = _resolve(scaler_path)
    destination = _resolve(output)
    zip_destination = Path(str(destination) + ".zip")
    summary_destination = Path(str(destination) + ".export.json")
    if destination.exists() or zip_destination.exists() or summary_destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing export: {destination} or {zip_destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v5-export-stage-", dir=str(destination.parent)))
    build_dir = stage_parent / "package"
    zip_stage = stage_parent / "package.zip"
    try:
        manifest, directory_check = _build_package(
            config_path, checkpoint_path, scaler_path, build_dir, precision,
            max_bytes=max_bytes, verify_process=verify_process)
        if directory_check["directory_bytes"] >= max_bytes:
            raise ValueError(f"Package directory must be < {max_bytes:,} bytes")
        _write_zip(build_dir, zip_stage)
        zip_check = _verify_zip(zip_stage, max_bytes, build_dir)
        if zip_check["zip_bytes"] >= max_bytes or zip_check["zip_uncompressed_bytes"] >= max_bytes:
            raise ValueError(f"Package ZIP and uncompressed contents must each be < {max_bytes:,} bytes")
        # Rename staged artifacts on the same filesystem. If publishing the ZIP
        # fails, roll back the just-published directory rather than leave a
        # misleading complete package behind.
        os.replace(build_dir, destination)
        try:
            os.replace(zip_stage, zip_destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        result = {
            "status": "passed", "package": str(destination.resolve()),
            "zip": str(zip_destination.resolve()),
            "package_sha256": hash_path(destination), "zip_sha256": sha256_file(zip_destination),
            "directory_bytes": directory_check["directory_bytes"], "zip_bytes": zip_check["zip_bytes"],
            "limit_bytes": int(max_bytes), "precision": "fp16",
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "manifest_file_count": len(manifest["files"]),
            "package_reload_verified": bool(manifest["package_reload_verification"].get("verified")),
            "package_reload_verification": manifest["package_reload_verification"],
            "verification": {**directory_check, **zip_check, "status": "passed"},
            "official_test_evaluated": False,
        }
        summary_stage = stage_parent / "export.json"
        _json_write(summary_stage, result)
        try:
            os.replace(summary_stage, summary_destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            zip_destination.unlink(missing_ok=True)
            raise
        return result
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def load_offline_predictor(package_path: str | Path, device: str = "cpu"):
    """Load the package runtime from an offline directory or verified ZIP."""
    path = _resolve(package_path)
    verify_package(path)
    if path.is_dir():
        package_dir = path
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="v5_predictor_")
        _safe_extract(path, Path(cleanup.name))
        package_dir = Path(cleanup.name)
    # Use package-local source to prove inference does not need the project
    # checkout. Load the generated runtime under a private module name.
    import importlib.util
    module_path = package_dir / "infer.py"
    module_name = f"_v5_offline_infer_{hash_path(package_dir)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        if cleanup is not None:
            cleanup.cleanup()
        raise ImportError("Cannot load packaged infer.py")
    package_parent = str(package_dir)
    inserted = package_parent not in sys.path
    if inserted:
        sys.path.insert(0, package_parent)
    runtime_names = ("v5_model", "v5_data", "aligned_dataset", "p2", "pipeline")
    previous_modules = {name: sys.modules.pop(name, None) for name in runtime_names}
    prior_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        predictor = module.OfflinePredictor(package_dir, device=device)
        predictor._temporary_directory = cleanup
        predictor._package_infer_module = module
        return predictor
    except Exception:
        if cleanup is not None:
            cleanup.cleanup()
        raise
    finally:
        sys.dont_write_bytecode = prior_bytecode_setting
        # The predictor and its methods retain references to these package-local
        # modules; restore the caller's project modules for later CLI work.
        for name in runtime_names:
            sys.modules.pop(name, None)
            if previous_modules[name] is not None:
                sys.modules[name] = previous_modules[name]
        if inserted:
            try:
                sys.path.remove(package_parent)
            except ValueError:
                pass


def _write_predictions_csv(predictions: dict, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(predictions["logits"], dtype=np.float64)
    intensity = np.asarray(predictions["intensity_raw"], dtype=np.float64)
    ids = np.asarray(predictions.get("sample_ids", [str(i) for i in range(len(logits))]), dtype=str)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "negative_probability", "neutral_probability",
                         "positive_probability", "predicted_class", "intensity_raw", "intensity_clipped"])
        for sid, prob, value in zip(ids, probs, intensity):
            writer.writerow([sid, *[float(x) for x in prob], int(prob.argmax()), float(value),
                             float(np.clip(value, -3.0, 3.0))])


def predict_attachment3(package_path: str | Path, folder: str | Path,
                        output: str | Path, device: str = "cpu") -> dict:
    predictor = load_offline_predictor(package_path, device=device)
    source_tokenizer = predictor.tokenizer_source
    special_ids = tuple(int(x) for x in (source_tokenizer.pad_token_id,
                                         source_tokenizer.cls_token_id, source_tokenizer.sep_token_id))
    dataset = AlignedDataset.from_attachment3(folder, special_token_ids=special_ids)
    samples = [dataset[i] for i in range(len(dataset))]
    predictions = predictor.predict_samples(samples)
    predictions["sample_ids"] = np.asarray([sample["sample_id"] for sample in samples], dtype=str)
    _write_predictions_csv(predictions, output)
    return {"status": "passed", "rows": len(samples), "output": str(Path(output).resolve()),
            "package_sha256": hash_path(package_path), "official_test_evaluated": False}


def freeze_model(config_path: str | Path, validation_path: str | Path,
                 package_path: str | Path, output: str | Path) -> dict:
    """Freeze a validation-selected package only after package reload is verified."""
    destination = _resolve(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite freeze record: {destination}")
    config = _json_read(_resolve(config_path))
    validation = _json_read(_resolve(validation_path))
    package = _resolve(package_path)
    verification = verify_package(package)
    digest = hash_path(package)
    if validation.get("official_test_evaluated", False):
        raise ValueError("Freeze requires validation-only metrics")
    if validation.get("package_sha256") != digest:
        raise ValueError("Validation metrics package_sha256 does not match the exported package")
    if validation.get("package_reload_verified") is not True:
        raise ValueError("Validation metrics must record package_reload_verified=true")
    if verification.get("package_reload_verified") is not True:
        raise ValueError("Package manifest does not contain a successful independent reload verification")
    complete = validation.get("complete")
    if not isinstance(complete, dict) or not {"accuracy", "macro_f1", "mae"}.issubset(complete):
        raise ValueError("Validation record must contain complete accuracy, macro_f1, and mae")
    missing_metrics = validation.get("missing_all_random")
    if not isinstance(missing_metrics, dict):
        raise ValueError("Validation record must contain missing_all_random metrics")
    manifest = {
        "status": "frozen", "config": _sensitive_scrub(config),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest(),
        "validation": validation,
        "package": str(package.resolve()), "package_sha256": digest,
        "package_zip": str(Path(str(package) + ".zip").resolve()) if package.is_dir() else str(package.resolve()),
        "package_zip_sha256": verification.get("zip_sha256"),
        "package_verification": verification,
        "package_reload_verification": _json_read(package / "manifest.json")["package_reload_verification"] if package.is_dir() else verification,
        "official_test_evaluated": False,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    _json_write(temporary, manifest)
    os.replace(temporary, destination)
    return manifest


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export and independently verify an offline FP16 student package")
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--scaler", required=True)
    export.add_argument("--output", required=True, help="new package directory; writes sibling .zip")
    export.add_argument("--precision", choices=("fp16",), default="fp16",
                        help="FP16 weight storage (INT8 is currently unsupported)")
    verify = sub.add_parser("verify", help="verify all package hashes and strict size limits")
    verify.add_argument("package")
    freeze = sub.add_parser("freeze", help="freeze a validation-selected, reload-verified package")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--validation", required=True)
    freeze.add_argument("--package", required=True)
    freeze.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "export":
        return export_package(args.config, args.checkpoint, args.scaler,
                              args.output, args.precision)
    if args.command == "verify":
        return verify_package(args.package)
    if args.command == "freeze":
        return freeze_model(args.config, args.validation, args.package, args.output)
    raise AssertionError(args.command)


if __name__ == "__main__":
    print(json.dumps(_cli(), ensure_ascii=False, indent=2))
