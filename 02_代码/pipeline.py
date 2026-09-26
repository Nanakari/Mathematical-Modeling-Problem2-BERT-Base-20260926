"""Train-only masked normalization and reproducible token-level corruption."""
import hashlib
import json
from pathlib import Path
import numpy as np

MODALITIES = ("text", "audio", "vision")


def clone(sample):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}


class MaskedStandardizer:
    def __init__(self):
        self.stats = {}

    def fit(self, dataset, *, split):
        if split != "train" or getattr(dataset, "source", "").split("/")[-1] != "train":
            raise ValueError("Fit requires the official training Dataset")
        if self.stats:
            raise ValueError("Already fitted; create a separate standardizer explicitly")
        accum = {m: [0, None, None] for m in ("audio", "vision")}
        for i in range(len(dataset)):
            sample = dataset[i]
            for m in accum:
                x = sample[m][sample[m + "_observed"]].astype(np.float64)
                if not len(x):
                    continue
                n, mean, m2 = accum[m]
                bn, bm = len(x), x.mean(0)
                bm2 = ((x-bm)**2).sum(0)
                if n == 0:
                    accum[m] = [bn, bm, bm2]
                else:
                    delta = bm-mean
                    accum[m] = [n+bn, mean+delta*bn/(n+bn), m2+bm2+delta**2*n*bn/(n+bn)]
        for m, (n, mean, m2) in accum.items():
            if n == 0:
                raise ValueError(f"No observed training rows for {m}")
            scale = np.sqrt(np.maximum(m2/n, 0))
            self.stats[m] = {"count": n, "mean": mean, "scale": np.where(scale > 1e-8, scale, 1.)}
        return self

    def transform(self, sample):
        if set(self.stats) != {"audio", "vision"}:
            raise ValueError("Standardizer not fitted")
        result = clone(sample)
        for m, stats in self.stats.items():
            mask = sample[m + "_observed"]
            x = np.zeros_like(sample[m], dtype=np.float32)
            x[mask] = ((sample[m][mask]-stats["mean"])/stats["scale"]).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Non-finite observed standardized input")
            result[m] = x
        return result

    def save(self, path):
        payload = {m: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()}
                   for m, s in self.stats.items()}
        Path(path).write_text(json.dumps({"fit_split": "train", "stats": payload}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["fit_split"] != "train":
            raise ValueError("Invalid fit provenance")
        obj = cls()
        obj.stats = {m: {k: np.asarray(v) if k in {"mean", "scale"} else v for k, v in s.items()}
                     for m, s in payload["stats"].items()}
        return obj


def mask_fingerprint(sample):
    h = hashlib.sha256(str(sample["sample_id"]).encode())
    for key in ("extent_mask", "text_pool_mask", "text_observed", "audio_observed", "vision_observed"):
        h.update(np.asarray(sample[key], dtype=np.uint8).tobytes())
    return h.hexdigest()


def nested_intervals(sample, modality, rates, position, seed):
    """Nested contiguous intervals in original coordinates; preserve observed info.

    The anchor is independent of rate. If constraints prevent expansion, keep the
    preceding interval and report its achieved size rather than break nesting.
    """
    if modality not in MODALITIES or position not in {"start", "middle", "end", "random"}:
        raise ValueError("Unsupported modality or position")
    if rates != sorted(set(rates)) or any(not 0 <= r < 1 for r in rates):
        raise ValueError("Rates must be unique ascending values in [0,1)")
    extent = sample["extent_mask"].astype(bool)
    eligible = extent.copy()
    if modality == "text":
        eligible &= sample["text_pool_mask"]
    observed = sample[modality + "_observed"].astype(bool)
    seed_bytes = f"{seed}|{sample['sample_id']}|{modality}|{position}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little"))
    locations = np.flatnonzero(eligible)
    anchor = float(rng.choice(locations)) if len(locations) else 0.
    if len(locations) and position != "random":
        anchor = {"start": float(locations[0]), "middle": float((locations[0]+locations[-1])/2),
                  "end": float(locations[-1])}[position]
    previous = None
    results = {}
    for rate in rates:
        budget = min(int(extent.sum()*rate), max(int(extent.sum())-1, 0))
        selected = previous
        for width in range(budget, 0, -1):
            candidates = []
            for start in range(len(extent)-width+1):
                end = start+width
                if not eligible[start:end].all():
                    continue
                if previous and not (start <= previous[0] and end >= previous[1]):
                    continue
                if observed.any() and observed[start:end].sum() >= observed.sum():
                    continue
                center = start if position == "start" else end-1 if position == "end" else (start+end-1)/2
                candidates.append((abs(center-anchor), start, end))
            if candidates:
                _, start, end = min(candidates)
                selected = [start, end]
                break
        results[str(rate)] = selected
        previous = selected
    return results


def make_plan(sample, modalities, rate, position, seed, *, rates=None):
    rates = sorted(set([rate] if rates is None else rates))
    if rate not in rates or not set(modalities).issubset(MODALITIES):
        raise ValueError("Invalid scenario")
    intervals = {m: nested_intervals(sample, m, rates, position, seed)[str(rate)] for m in modalities}
    n = int(sample["extent_mask"].sum())
    return {"sample_id": sample["sample_id"], "mask_fingerprint": mask_fingerprint(sample),
            "seed": seed, "position": position, "requested_rate": rate,
            "intervals": intervals, "valid_positions": n,
            "actual_rates": {m: (b-a)/n if span and n else 0. for m, span in intervals.items()
                             for a,b in ([span] if span else [(0,0)])}}


def apply_plan(sample, plan):
    if plan["mask_fingerprint"] != mask_fingerprint(sample) or plan["sample_id"] != sample["sample_id"]:
        raise ValueError("Plan does not match sample ID/masks")
    result = clone(sample)
    for m in MODALITIES:
        missing = np.zeros_like(sample["extent_mask"])
        span = plan["intervals"].get(m)
        if span:
            start, end = span
            if not 0 <= start < end <= len(missing) or not sample["extent_mask"][start:end].all():
                raise ValueError("Interval outside valid extent")
            missing[start:end] = True
        result[m + "_missing"] = missing
        result[m + "_observed"] &= ~missing
        if m == "text":
            # Remove information BEFORE encoding. Keep extent and other modalities.
            result["input_ids"][missing] = 0
            result["attention_mask"][missing] = 0
            result["token_type_ids"][missing] = 0
            result["text_pool_mask"] &= ~missing
        else:
            result[m][missing] = 0
    return result


class PreparedDataset:
    def __init__(self, dataset, standardizer, *, plans=None, augment=False, seed=0,
                 augmentation_rate=.3, modalities=MODALITIES, position="random"):
        if plans is not None and augment:
            raise ValueError("Choose fixed plans or training augmentation")
        if augment and getattr(dataset, "source", "").split("/")[-1] != "train":
            raise ValueError("Random augmentation is training-only")
        self.dataset, self.standardizer = dataset, standardizer
        self.plans, self.augment, self.seed, self.epoch = plans, augment, seed, 0
        self.augmentation_rate, self.modalities, self.position = augmentation_rate, modalities, position

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.plans is not None:
            plan = self.plans[sample["sample_id"]]
        elif self.augment:
            plan = make_plan(sample, self.modalities, self.augmentation_rate, self.position, self.seed+self.epoch)
        else:
            plan = make_plan(sample, (), 0., "start", self.seed)
        return self.standardizer.transform(apply_plan(sample, plan))
