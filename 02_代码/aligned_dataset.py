"""Aligned token-input Dataset shared by attachment 2 and attachment 3.

Map-style (__len__, __getitem__) protocol, with NumPy values supported by
PyTorch's default_collate. No torch dependency or model download at import time.
Only read trusted competition pickle files.
"""
from pathlib import Path
import pickle
import numpy as np


class AlignedDataset:
    def __init__(self, part, *, labeled, source="attachment2", special_token_ids=()):
        self.part = part
        self.labeled = bool(labeled)
        self.source = str(source)
        # Set only after confirming tokenizer special-token definitions.
        self.special_token_ids = tuple(special_token_ids)
        self._validate()

    @classmethod
    def from_attachment2(cls, path, split, **kwargs):
        if split not in {"train", "valid", "test"}:
            raise ValueError("Expected supplied train/valid/test split")
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        # Keep only the selected split and required arrays; discard large text.
        return cls(cls._select(data[split]), labeled=True,
                   source=f"attachment2/{split}", **kwargs)

    @classmethod
    def from_attachment3(cls, folder, **kwargs):
        files = sorted(Path(folder).glob("*.pkl"))
        if not files:
            raise ValueError("No pickle files in selected aligned folder")
        parts, ids = [], []
        for path in files:
            with path.open("rb") as f:
                data = pickle.load(f)
            p = cls._select(data["test"])
            # Explicitly reject unaligned schemas rather than fill missing fields.
            cls(p, labeled=False, source=path.name, **kwargs)
            parts.append(p)
            n = len(p["text_bert"])
            ids.extend(f"{path.name}#{i}" for i in range(n))
        merged = {k: np.concatenate([p[k] for p in parts])
                  for k in ("text_bert", "audio", "vision")}
        merged["id"] = ids  # Traceability keys, NOT confirmed submission IDs.
        return cls(merged, labeled=False, source="attachment3/aligned", **kwargs)

    @staticmethod
    def _select(part):
        return {k: v for k, v in part.items() if k in {
            "id", "text_bert", "audio", "vision", "classification_labels", "regression_labels"}}

    def _validate(self):
        p = self.part
        for key in ("text_bert", "audio", "vision"):
            if key not in p:
                raise ValueError(f"{self.source}: missing {key}; do not substitute zeros")
        b = np.asarray(p["text_bert"])
        if b.ndim != 3 or b.shape[1:] != (3, 50):
            raise ValueError("Expected text_bert (N,3,50)")
        n = len(b)
        if not np.isfinite(b).all() or not np.equal(b, np.floor(b)).all() or (b < 0).any():
            raise ValueError("Token inputs must be finite, nonnegative integers")
        if not np.isin(b[:, 1], [0, 1]).all():
            raise ValueError("Attention must be binary")
        # Current files have intact, contiguous attention metadata. Fail rather
        # than infer a common timeline from future corrupted attention fields.
        if np.any(np.diff(b[:, 1], axis=1) > 0):
            raise ValueError("Non-prefix attention: common extent needs explicit metadata")
        if not np.array_equal(b[:, 0] != 0, b[:, 1].astype(bool)):
            raise ValueError("Token/attention inconsistency: inspect corruption semantics")
        for m, dim in (("audio", 74), ("vision", 35)):
            x = np.asarray(p[m])
            if x.shape != (n, 50, dim) or not np.isfinite(x).all():
                raise ValueError(f"{m}: invalid aligned shape or non-finite input")
            observed = np.any(x != 0, axis=-1)
            if np.any(observed & ~b[:, 1].astype(bool)):
                raise ValueError(f"{m}: evidence outside candidate common extent; inspect alignment")
        if "id" in p and len(p["id"]) != n:
            raise ValueError("ID count mismatch")
        if self.labeled:
            for key in ("classification_labels", "regression_labels"):
                if key not in p or np.asarray(p[key]).shape != (n,):
                    raise ValueError(f"Invalid labels: {key}")
            c, r = np.asarray(p["classification_labels"]), np.asarray(p["regression_labels"])
            if not np.isin(c, [0, 1, 2]).all() or not np.isfinite(r).all() or (abs(r) > 3).any():
                raise ValueError("Labels out of range")
            if not np.array_equal(c, np.sign(r) + 1):
                raise ValueError("Class/sign mapping differs from verified 0/1/2 mapping")

    def __len__(self):
        return len(self.part["text_bert"])

    def __getitem__(self, index):
        p = self.part
        tokens = np.asarray(p["text_bert"][index], dtype=np.int64).copy()
        ids, attention, segments = tokens
        extent = attention.astype(bool)  # Candidate extent BEFORE augmentation.
        text_observed = extent & (ids != 0)
        text_pool = text_observed & ~np.isin(ids, self.special_token_ids)
        audio = np.asarray(p["audio"][index], dtype=np.float32).copy()
        vision = np.asarray(p["vision"][index], dtype=np.float32).copy()
        # Determine availability BEFORE standardization. Zero is unavailable,
        # not evidence of the specific cause (padding, extraction failure, etc.).
        audio_observed = extent & np.any(audio != 0, axis=-1)
        vision_observed = extent & np.any(vision != 0, axis=-1)
        sample_id = str(p["id"][index]) if "id" in p else f"{self.source}#{index}"
        result = {
            "sample_id": sample_id,
            "input_ids": ids,
            "attention_mask": attention,
            "token_type_ids": segments,
            "audio": audio,
            "vision": vision,
            "extent_mask": extent.copy(),
            "text_observed": text_observed,
            "text_pool_mask": text_pool,
            "audio_observed": audio_observed,
            "vision_observed": vision_observed,
            "has_label": self.labeled,
        }
        if self.labeled:
            result["class_label"] = np.int64(p["classification_labels"][index])
            result["regression_label"] = np.float32(p["regression_labels"][index])
        # No dummy neutral label for unlabeled data. Keep labeled/unlabeled
        # examples in separate DataLoaders with homogeneous dictionary keys.
        return result
