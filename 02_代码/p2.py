"""Independent problem-2 components. All mask assumptions are explicit.

Only load the competition's trusted pickle files. Never overwrite source data.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

MODALITIES = ("text", "audio", "vision")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "E题" / "E题数据"


def load_features(version="aligned_50.pkl"):
    with (DATA / "附件2-数据集特征文件" / version).open("rb") as handle:
        return pickle.load(handle)


def stable_seed(seed, sample_id, modality=""):
    token = f"{seed}|{sample_id}|{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def candidate_masks(part, version="aligned_50.pkl"):
    """Return validity and availability separately; these are provisional policies.

    Aligned: use BERT attention as a candidate common valid-position mask.
    Unaligned: audio/vision require explicit lengths. Missing metadata fails closed.
    All-zero rows are excluded from availability, not declared proven corruption.
    Special BERT tokens are retained until temporal correspondence is clarified.
    """
    if "text" not in part or "text_bert" not in part:
        raise ValueError("Continuous text features and text_bert attention metadata required")
    attention = np.asarray(part["text_bert"])[:, 1, :]
    if not np.all(np.isin(attention, [0, 1])):
        raise ValueError("Attention mask is not binary")
    valid, available = {}, {}
    for m in MODALITIES:
        x = np.asarray(part[m])
        if version == "aligned_50.pkl" or m == "text":
            v = attention.astype(bool).copy()
        else:
            key = {"audio": "audio_lengths", "vision": "vision_lengths"}[m]
            if key not in part:
                raise ValueError(f"Missing length metadata: {key}")
            lengths = np.asarray(part[key]).reshape(-1)
            if np.any(lengths < 0) or np.any(lengths > x.shape[1]):
                raise ValueError(f"Invalid lengths: {key}")
            v = np.arange(x.shape[1])[None, :] < lengths[:, None]
        if v.shape != x.shape[:2]:
            raise ValueError(f"Mask shape mismatch: {m}")
        valid[m] = v
        available[m] = v & np.isfinite(x).all(-1) & np.any(x != 0, axis=-1)
    return valid, available


def _spans(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def corrupt(features, valid, available, ids, modalities=("text",), rate=0.3,
            position="random", seed=0, segments=1, synchronized=False):
    """Mask continuous intervals, measured against valid positions.

    Intervals are half-open. One valid position remains per modality. A modality
    with available input retains at least one available position. Existing
    unavailable rows count toward selected interval length but not newly lost data.
    Multiple segments are disjoint; actual rates are recorded when exact budgets
    cannot be met. Synchronized mode requires identical aligned valid masks.
    """
    if not 0 <= rate < 1 or segments < 1:
        raise ValueError("Use 0 <= rate < 1 and segments >= 1 for local missingness")
    if position not in {"start", "middle", "end", "random"}:
        raise ValueError(position)
    if not modalities or not set(modalities).issubset(MODALITIES):
        raise ValueError("Unknown or empty modality selection")
    if synchronized:
        first = valid[modalities[0]]
        if any(not np.array_equal(first, valid[m]) for m in modalities):
            raise ValueError("Synchronized masking requires shared aligned coordinates")
    out = {m: np.array(x, copy=True) for m, x in features.items()}
    observed = {m: a.copy() for m, a in available.items()}
    missing = {m: np.zeros_like(v) for m, v in valid.items()}
    records = []
    for i, sid in enumerate(ids):
        for m in modalities:
            rng = np.random.default_rng(stable_seed(seed, sid, "sync" if synchronized else m))
            n = int(valid[m][i].sum())
            budget = min(int(np.floor(n * rate)), max(n - 1, 0))
            remaining = valid[m][i].copy()
            intervals = []
            for j in range(segments):
                target = (budget + segments - j - 1) // (segments - j)
                if target <= 0:
                    break
                spans = _spans(remaining)
                if not spans:
                    break
                width = min(target, max(b - a for a, b in spans))
                protect = modalities if synchronized else (m,)
                starts = np.array([], dtype=int)
                while width > 0:
                    candidates = [s for a, b in spans for s in range(a, b - width + 1)]
                    accepted = []
                    for s in candidates:
                        safe = True
                        for protected in protect:
                            remaining_available = available[protected][i] & ~missing[m][i]
                            count = int(remaining_available.sum())
                            if count and int(remaining_available[s:s+width].sum()) >= count:
                                safe = False
                                break
                        if safe:
                            accepted.append(s)
                    starts = np.asarray(accepted, dtype=int)
                    if len(starts):
                        break
                    width -= 1
                if width == 0:
                    break
                if position == "random":
                    start = int(rng.choice(starts))
                elif position == "start":
                    start = int(starts[0])
                elif position == "end":
                    start = int(starts[-1])
                else:
                    center = np.flatnonzero(valid[m][i]).mean()
                    start = int(starts[np.argmin(abs(starts + (width - 1) / 2 - center))])
                end = start + width
                missing[m][i, start:end] = True
                remaining[start:end] = False
                intervals.append([start, end])
                budget -= width
            selected = missing[m][i]
            newly_lost = int((selected & available[m][i]).sum())
            observed[m][i, selected] = False
            out[m][i, selected] = 0
            records.append({"id": str(sid), "modality": m, "intervals": intervals,
                            "valid_positions": n, "requested_rate": rate,
                            "masked_positions": int(selected.sum()),
                            "actual_rate": float(selected.sum() / n) if n else 0.0,
                            "newly_lost_available_positions": newly_lost,
                            "remaining_available_positions": int(observed[m][i].sum())})
    return out, observed, missing, records


def pooled_features(features, observed):
    blocks = []
    for m in MODALITIES:
        x, mask = features[m], observed[m]
        safe = np.where(mask[..., None], x, 0.0)
        if not np.isfinite(safe).all():
            raise ValueError("Observed feature contains non-finite values")
        blocks.append(safe.sum(1) / np.maximum(mask.sum(1, keepdims=True), 1))
    return np.concatenate(blocks, axis=1)


def metrics(y_class, y_reg, pred_class, pred_reg):
    y_reg, pred_reg = np.asarray(y_reg).ravel(), np.asarray(pred_reg).ravel()
    pearson = None
    if len(y_reg) > 1 and np.std(y_reg) > 0 and np.std(pred_reg) > 0:
        pearson = float(np.corrcoef(y_reg, pred_reg)[0, 1])
    return {"accuracy": float(accuracy_score(y_class, pred_class)),
            "f1_macro": float(f1_score(y_class, pred_class, labels=[0, 1, 2], average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_class, pred_class, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "f1_per_class": f1_score(y_class, pred_class, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "mae": float(mean_absolute_error(y_reg, pred_reg)), "pearson": pearson}


class TinyMaskedModel:
    """NumPy trainable projection + masked mean + concatenation + dual heads.

    Deliberately lightweight engineering candidate, not a temporal encoder or a
    claim of robustness. No contextual reconstruction. Exact analytic gradients.
    """
    def __init__(self, dimensions, hidden=12, seed=0):
        rng = np.random.default_rng(seed)
        self.parameters = {}
        for m, dim in dimensions.items():
            self.parameters[m] = rng.normal(0, 1 / np.sqrt(dim), (dim, hidden))
        self.parameters["class"] = rng.normal(0, 0.1, (hidden * 3, 3))
        self.parameters["reg"] = rng.normal(0, 0.1, (hidden * 3, 1))
        self.parameters["class_bias"] = np.zeros(3)
        self.parameters["reg_bias"] = np.zeros(1)

    def forward(self, features, observed):
        cache, blocks = {}, []
        for m in MODALITIES:
            a = observed[m][..., None]
            x = np.where(a, features[m], 0.0)
            h = np.tanh(x @ self.parameters[m])
            count = np.maximum(a.sum(1), 1)
            blocks.append((h * a).sum(1) / count)
            cache[m] = (x, h, a, count)
        z = np.concatenate(blocks, 1)
        logits = z @ self.parameters["class"] + self.parameters["class_bias"]
        exp = np.exp(logits - logits.max(1, keepdims=True))
        prob = exp / exp.sum(1, keepdims=True)
        raw = (z @ self.parameters["reg"] + self.parameters["reg_bias"]).ravel()
        intensity = 3 * np.tanh(raw)
        return prob, intensity, (z, cache)

    def loss_grad(self, features, observed, y_class, y_reg, class_weight=1., reg_weight=1.):
        prob, intensity, (z, cache) = self.forward(features, observed)
        n = len(y_class)
        ce = -np.log(np.maximum(prob[np.arange(n), y_class], 1e-15)).mean()
        error = intensity - y_reg
        mse = np.mean(error ** 2)
        dl = prob.copy()
        dl[np.arange(n), y_class] -= 1
        dl *= class_weight / n
        dr = (reg_weight * 2 / n * error * 3 * (1 - (intensity / 3) ** 2))[:, None]
        grads = {"class": z.T @ dl, "reg": z.T @ dr,
                 "class_bias": dl.sum(0), "reg_bias": dr.sum(0)}
        dz = dl @ self.parameters["class"].T + dr @ self.parameters["reg"].T
        for m, block in zip(MODALITIES, np.split(dz, 3, axis=1)):
            x, h, a, count = cache[m]
            dh = block[:, None, :] / count[:, None, :] * a * (1 - h ** 2)
            grads[m] = np.einsum("btd,bth->dh", x, dh)
        return float(class_weight * ce + reg_weight * mse), grads
