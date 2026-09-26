"""Measure realized local missing length in aligned positions on validation only."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/bert_base_vs_minilm_20260926"))

from v5_data import load_train_valid, make_missing_view
from severity_grid import grid_cases


def main():
    _, valid = load_train_valid(ROOT / "outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl")
    samples = [valid[i] for i in range(len(valid))]
    rows = []
    for case in grid_cases()[1:]:
        removed, fraction, extent = [], [], []
        for sample in samples:
            masked = make_missing_view(sample, case["modalities"], case["rate"],
                                       case["position"], case["seed"],
                                       nested_rates=(0.1, 0.3, 0.5, 0.7))
            candidate = int(np.sum(sample["extent_mask"]))
            for modality in case["modalities"]:
                before = np.asarray(sample[f"{modality}_observed"], dtype=bool)
                after = np.asarray(masked[f"{modality}_observed"], dtype=bool)
                count = int(np.sum(before & ~after))
                removed.append(count)
                fraction.append(count / candidate if candidate else 0.0)
                extent.append(candidate)
        rows.append({"case": case["name"], "subset": "+".join(case["modalities"]),
                     "position": case["position"], "nominal_rate": case["rate"],
                     "mean_removed_observed_slots": float(np.mean(removed)),
                     "mean_realized_fraction_of_extent": float(np.mean(fraction)),
                     "mean_extent_slots": float(np.mean(extent)),
                     "min_removed_slots": int(np.min(removed)),
                     "max_removed_slots": int(np.max(removed))})
    out = LANE / "realized_missing_lengths.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
