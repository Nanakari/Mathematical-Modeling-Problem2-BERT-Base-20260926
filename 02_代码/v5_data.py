"""Train/validation-only data preparation and leak-safe visible-text encoding."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer

from aligned_dataset import AlignedDataset
from p2 import DATA
from pipeline import MaskedStandardizer, apply_plan, make_plan


EXPECTED_SPLITS = {"train": 3395, "valid": 728}
SOURCE_SPECIAL_IDS = (0, 100, 101, 102)


def load_train_valid(path: str | Path | None = None, *, pickle_loader: Callable | None = None,
                     strict_sizes: bool = True):
    """Load official train/valid arrays and discard the test object immediately.

    Pickle deserialization necessarily reads the enclosing object. This function
    never indexes or returns its ``test`` split; it removes that key before
    touching train/valid fields. Only trusted competition pickles are supported.
    ``pickle_loader`` is injectable so tests can assert split access behavior.
    """
    feature_path = Path(path) if path is not None else DATA / "附件2-数据集特征文件" / "aligned_50.pkl"
    loader = pickle_loader or pickle.load
    with feature_path.open("rb") as handle:
        raw = loader(handle)
    if not isinstance(raw, dict):
        raise TypeError("Expected a split dictionary in the aligned feature pickle")
    raw.pop("test", None)
    if "train" not in raw or "valid" not in raw:
        raise ValueError("The aligned feature file must contain official train and valid splits")
    train_part, valid_part = raw["train"], raw["valid"]
    train = AlignedDataset(AlignedDataset._select(train_part), labeled=True,
                           source="attachment2/train", special_token_ids=(101, 102))
    valid = AlignedDataset(AlignedDataset._select(valid_part), labeled=True,
                           source="attachment2/valid", special_token_ids=(101, 102))
    del raw, train_part, valid_part
    if strict_sizes:
        for name, dataset in (("train", train), ("valid", valid)):
            expected = EXPECTED_SPLITS[name]
            if len(dataset) != expected:
                raise ValueError(f"Expected fixed official {name} split size {expected}, got {len(dataset)}")
        train_ids = {train[i]["sample_id"] for i in range(len(train))}
        valid_ids = {valid[i]["sample_id"] for i in range(len(valid))}
        overlap = train_ids.intersection(valid_ids)
        if overlap:
            raise ValueError(f"Train/valid sample IDs overlap: {len(overlap)}")
    return train, valid


def load_attachment3(folder: str | Path):
    """Load only aligned, unlabeled attachment 3 arrays in deterministic order."""
    return AlignedDataset.from_attachment3(folder, special_token_ids=(101, 102))


def load_tokenizers(backbone_dir: str | Path, source_vocab: str | Path | None = None):
    backbone_dir = Path(backbone_dir)
    vocab = Path(source_vocab) if source_vocab else Path(__file__).resolve().parent / "models" / "bert_mini" / "vocab.txt"
    source = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    target = AutoTokenizer.from_pretrained(str(backbone_dir), local_files_only=True, use_fast=True)
    return source, target


def _same_wordpiece_vocab(source_tokenizer, target_tokenizer) -> bool:
    try:
        source_vocab = source_tokenizer.get_vocab()
        target_vocab = target_tokenizer.get_vocab()
    except (AttributeError, TypeError):
        return False
    if source_vocab != target_vocab:
        return False
    for name in ("pad_token_id", "unk_token_id", "cls_token_id", "sep_token_id"):
        if getattr(source_tokenizer, name, None) != getattr(target_tokenizer, name, None):
            return False
    return True


def _runs(positions: np.ndarray) -> list[np.ndarray]:
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) != 1) + 1
    return [part for part in np.split(positions, cuts) if len(part)]


def _encode_surface_tokens(tokens: list[str], positions: list[int], source_tokenizer,
                           target_tokenizer):
    """Retokenize one visible source-token island without consulting raw_text."""
    out_ids: list[int] = []
    out_pool: list[bool] = []
    out_spans: list[list[int]] = []
    group_tokens: list[str] = []
    group_positions: list[int] = []

    def flush_group():
        if not group_tokens:
            return
        surface = source_tokenizer.convert_tokens_to_string(group_tokens).strip()
        if surface:
            encoded = target_tokenizer(surface, add_special_tokens=False,
                                       truncation=False, return_attention_mask=False)
            ids = list(encoded["input_ids"])
            out_ids.extend(int(x) for x in ids)
            out_pool.extend([True] * len(ids))
            span = [int(min(group_positions)), int(max(group_positions) + 1)]
            out_spans.extend([span.copy() for _ in ids])
        group_tokens.clear()
        group_positions.clear()

    for token, position in zip(tokens, positions):
        if token == source_tokenizer.unk_token:
            flush_group()
            unk_id = target_tokenizer.unk_token_id
            if unk_id is None:
                raise ValueError("Target tokenizer has no unknown-token ID")
            out_ids.append(int(unk_id))
            # UNK means an observed source token could not be represented by
            # the target vocabulary; it remains evidence for pooling/KD.
            out_pool.append(True)
            out_spans.append([int(position), int(position + 1)])
        else:
            group_tokens.append(token)
            group_positions.append(int(position))
    flush_group()
    return out_ids, out_pool, out_spans


def encode_visible_text(sample: dict, source_tokenizer, target_tokenizer,
                        max_length: int = 128, *, vocab_compatible: bool | None = None
                        ) -> dict[str, np.ndarray | list]:
    """Encode only visible BERT tokens and preserve missing islands as boundaries.

    If the candidate tokenizer has the exact verified BERT WordPiece vocabulary,
    the supplied 50-position IDs/masks are retained verbatim. Otherwise, each
    contiguous visible island is reconstructed from its visible WordPiece IDs,
    retokenized independently, and separated by a target SEP token. The
    ``source_spans`` array maps each new token to its source island extent for
    diagnostics; it is deliberately not used for MAG-style positionwise fusion.
    """
    if max_length < 2:
        raise ValueError("max_length must allow target start/end special tokens")
    raw_ids = np.asarray(sample["input_ids"], dtype=np.int64)
    raw_attn = np.asarray(sample["attention_mask"], dtype=bool)
    observed = np.asarray(sample["text_observed"], dtype=bool) & raw_attn
    pool = np.asarray(sample["text_pool_mask"], dtype=bool) & observed
    segments = np.asarray(sample["token_type_ids"], dtype=np.int64)
    if raw_ids.shape != raw_attn.shape or raw_ids.shape != observed.shape:
        raise ValueError("Text IDs and masks must have identical sequence length")
    compatible = (_same_wordpiece_vocab(source_tokenizer, target_tokenizer)
                  if vocab_compatible is None else bool(vocab_compatible))
    if compatible:
        ids = raw_ids.copy()
        attn = observed.copy()
        seg = segments.copy()
        text_pool = pool.copy()
        spans = np.full((len(ids), 2), -1, dtype=np.int64)
        for index in np.flatnonzero(pool):
            spans[index] = (index, index + 1)
        if len(ids) > max_length:
            # Keep the original leading position and the original final
            # candidate SEP, while truncating only interior/right content.
            active_extent = np.flatnonzero(raw_attn)
            final_index = int(active_extent[-1]) if len(active_extent) else len(ids) - 1
            ids = np.concatenate([ids[:max_length - 1], ids[final_index:final_index + 1]])
            attn = np.concatenate([attn[:max_length - 1], attn[final_index:final_index + 1]])
            seg = np.concatenate([seg[:max_length - 1], seg[final_index:final_index + 1]])
            text_pool = np.concatenate([text_pool[:max_length - 1], [False]])
            spans = np.concatenate([spans[:max_length - 1], [[-1, -1]]], axis=0)
    else:
        content_positions = np.flatnonzero(pool)
        content_ids: list[int] = []
        content_pool: list[bool] = []
        content_spans: list[list[int]] = []
        prior_end = None
        for run in _runs(content_positions):
            if prior_end is not None:
                sep_id = target_tokenizer.sep_token_id
                if sep_id is None:
                    raise ValueError("Target tokenizer needs a separator token to preserve missing gaps")
                content_ids.append(int(sep_id))
                content_pool.append(False)
                content_spans.append([int(prior_end), int(run[0])])
            run_ids = raw_ids[run].tolist()
            source_tokens = source_tokenizer.convert_ids_to_tokens(run_ids)
            encoded_ids, encoded_pool, encoded_spans = _encode_surface_tokens(
                source_tokens, run.tolist(), source_tokenizer, target_tokenizer)
            content_ids.extend(encoded_ids)
            content_pool.extend(encoded_pool)
            content_spans.extend(encoded_spans)
            prior_end = int(run[-1]) + 1
        if not content_ids:
            unk = target_tokenizer.unk_token_id
            if unk is None:
                unk = target_tokenizer.pad_token_id or 0
            content_ids = [int(unk)]
            content_pool = [False]
            content_spans = [[-1, -1]]
        full_ids = list(target_tokenizer.build_inputs_with_special_tokens(content_ids))
        # Fast HF tokenizers reject get_special_tokens_mask(..., False). Build
        # the same wrapper template around unique sentinel IDs to locate only
        # the outer added-special slots. This avoids misclassifying content
        # [UNK]/[SEP] IDs as outer template tokens.
        sentinels = [-1_000_000 - index for index in range(len(content_ids))]
        template = list(target_tokenizer.build_inputs_with_special_tokens(sentinels))
        if len(template) != len(full_ids):
            raise ValueError("Tokenizer special-token template length differs from built sequence")
        full_pool = [False] * len(full_ids)
        full_spans = [[-1, -1] for _ in full_ids]
        for content_index, sentinel in enumerate(sentinels):
            positions = [i for i, token_id in enumerate(template) if token_id == sentinel]
            if len(positions) != 1:
                raise ValueError("Tokenizer did not preserve unique content markers in special-token template")
            target_index = positions[0]
            full_pool[target_index] = bool(content_pool[content_index])
            full_spans[target_index] = content_spans[content_index]
        if len(full_ids) > max_length:
            # Keep the initial special token and a final separator; preserve
            # ordered visible islands and never join across a missing interval.
            full_ids = full_ids[:max_length - 1] + [full_ids[-1]]
            full_pool = full_pool[:max_length - 1] + [False]
            full_spans = full_spans[:max_length - 1] + [[-1, -1]]
        ids = np.full(max_length, int(target_tokenizer.pad_token_id or 0), dtype=np.int64)
        attn = np.zeros(max_length, dtype=bool)
        seg = np.zeros(max_length, dtype=np.int64)
        text_pool = np.zeros(max_length, dtype=bool)
        spans = np.full((max_length, 2), -1, dtype=np.int64)
        count = min(len(full_ids), max_length)
        ids[:count] = full_ids[:count]
        attn[:count] = True
        text_pool[:count] = full_pool[:count]
        spans[:count] = np.asarray(full_spans[:count], dtype=np.int64)
    if len(ids) < max_length:
        pad = max_length - len(ids)
        ids = np.pad(ids, (0, pad), constant_values=int(target_tokenizer.pad_token_id or 0))
        attn = np.pad(attn, (0, pad), constant_values=False)
        seg = np.pad(seg, (0, pad), constant_values=0)
        text_pool = np.pad(text_pool, (0, pad), constant_values=False)
        spans = np.pad(spans, ((0, pad), (0, 0)), constant_values=-1)
    # Defend against tokenizer bugs or malformed input where pooling could see
    # a masked/padded token.
    text_pool &= attn
    return {
        "input_ids": ids.astype(np.int64, copy=False),
        "attention_mask": attn.astype(bool, copy=False),
        "token_type_ids": seg.astype(np.int64, copy=False),
        "text_pool_mask": text_pool.astype(bool, copy=False),
        "source_spans": spans,
        "vocab_compatible": bool(compatible),
    }


def make_missing_view(sample: dict, modalities, rate: float, position: str,
                      seed: int, *, nested_rates=(0.1, 0.3, 0.5)) -> dict:
    """Apply local continuous missing spans before any text encoding."""
    rates = sorted(set(float(x) for x in nested_rates))
    if rate not in rates:
        rates = sorted(set(rates + [float(rate)]))
    plan = make_plan(sample, tuple(modalities), float(rate), position, int(seed), rates=rates)
    return apply_plan(sample, plan)


def collate_samples(samples, standardizer: MaskedStandardizer | None,
                    source_tokenizer, target_tokenizer, max_text_length: int = 128,
                    *, device: torch.device | str | None = None, feature_clip: float = 5.0):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    transformed = [standardizer.transform(sample) if standardizer is not None else sample
                   for sample in samples]
    compatible = _same_wordpiece_vocab(source_tokenizer, target_tokenizer)
    text = [encode_visible_text(sample, source_tokenizer, target_tokenizer, max_text_length,
                                vocab_compatible=compatible)
            for sample in transformed]
    batch = {
        "text_ids": torch.as_tensor(np.stack([item["input_ids"] for item in text]), dtype=torch.long),
        "text_mask": torch.as_tensor(np.stack([item["attention_mask"] for item in text]), dtype=torch.bool),
        "text_pool_mask": torch.as_tensor(np.stack([item["text_pool_mask"] for item in text]), dtype=torch.bool),
        "text_segments": torch.as_tensor(np.stack([item["token_type_ids"] for item in text]), dtype=torch.long),
        "text_source_spans": torch.as_tensor(np.stack([item["source_spans"] for item in text]), dtype=torch.long),
    }
    for modality, dim in (("audio", 74), ("vision", 35)):
        values = np.stack([np.asarray(sample[modality], dtype=np.float32) for sample in transformed])
        mask = np.stack([np.asarray(sample[modality + "_observed"], dtype=bool) for sample in transformed])
        values = np.clip(values, -float(feature_clip), float(feature_clip))
        values[~mask] = 0.0
        if values.shape[1:] != (50, dim):
            raise ValueError(f"Expected {modality} input shape (B,50,{dim}), got {values.shape}")
        batch[modality] = torch.as_tensor(values, dtype=torch.float32)
        batch[modality + "_mask"] = torch.as_tensor(mask, dtype=torch.bool)
    if all("class_label" in sample for sample in samples):
        batch["class_label"] = torch.as_tensor([int(sample["class_label"]) for sample in samples], dtype=torch.long)
        batch["regression_label"] = torch.as_tensor(
            [float(sample["regression_label"]) for sample in samples], dtype=torch.float32)
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    if device is not None:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
    return batch


def fit_train_standardizer(train_dataset: AlignedDataset) -> MaskedStandardizer:
    return MaskedStandardizer().fit(train_dataset, split="train")


def split_fingerprint(dataset: AlignedDataset) -> dict:
    """Hash only official train or valid feature/target fields; never test."""
    if dataset.source not in {"attachment2/train", "attachment2/valid"}:
        raise ValueError("Only official train or valid may be fingerprinted here")
    h = hashlib.sha256()
    for key in ("id", "text_bert", "audio", "vision", "classification_labels", "regression_labels"):
        if key not in dataset.part:
            continue
        value = np.asarray(dataset.part[key])
        h.update(key.encode("utf-8"))
        if value.dtype.kind in "USO":
            h.update(json.dumps(value.tolist(), ensure_ascii=False, default=str).encode("utf-8"))
        else:
            h.update(np.ascontiguousarray(value).view(np.uint8))
    return {"split": dataset.source.rsplit("/", 1)[-1], "count": len(dataset), "sha256": h.hexdigest()}
