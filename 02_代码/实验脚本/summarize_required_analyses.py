"""Reuse frozen BERT-Base outputs for the requested problem-2 analyses."""
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
OLD = ROOT / "outputs/bert_base_vs_minilm_20260926"
RERUN = ROOT / "outputs/two_model_rerun_20260926"
ATTACH = ROOT.parent / "E题" / "E题数据" / "附件3-模态缺失特征样本" / "对齐版本"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grid():
    record = json.loads((OLD / "grid_student_bert_base_retest_seed1729.json").read_text(encoding="utf-8"))
    assert record["protocol"]["cell_count"] == 113
    assert record["protocol"]["official_test_evaluated"] is False
    base = record["cells"]["complete"]
    with (LANE / "realized_missing_lengths.csv").open(newline="", encoding="utf-8-sig") as handle:
        realized = {row["case"]: row for row in csv.DictReader(handle)}
    rows = []
    for case, metrics in record["cells"].items():
        if case == "complete":
            subset, position, rate = "none", "none", 0.0
        else:
            subset, position, text_rate = case.split("|")
            rate = float(text_rate)
        length = realized.get(case)
        rows.append({"case": case, "missing_modalities": subset, "position": position,
                     "nominal_rate": rate,
                     "mean_realized_fraction_of_extent":
                         float(length["mean_realized_fraction_of_extent"]) if length else 0.0,
                     "n": metrics["n"], "accuracy": metrics["accuracy"],
                     "macro_f1": metrics["macro_f1"], "mae": metrics["mae"],
                     "pearson": metrics["pearson"],
                     "accuracy_drop_from_complete": base["accuracy"] - metrics["accuracy"],
                     "macro_f1_drop_from_complete": base["macro_f1"] - metrics["macro_f1"],
                     "mae_increase_from_complete": metrics["mae"] - base["mae"],
                     "negative_recall": metrics["negative_recall"],
                     "neutral_recall": metrics["neutral_recall"],
                     "positive_recall": metrics["positive_recall"]})
    assert len(rows) == 113
    write_csv(LANE / "bert_base_missing_grid_metrics.csv", rows)
    grouped = []
    for subset in ("text", "audio", "vision", "text+audio", "text+vision",
                   "audio+vision", "text+audio+vision"):
        for rate in (0.1, 0.3, 0.5, 0.7):
            cells = [row for row in rows if row["missing_modalities"] == subset
                     and row["nominal_rate"] == rate]
            assert len(cells) == 4
            grouped.append({"missing_modalities": subset, "nominal_rate": rate,
                            "positions": 4,
                            "mean_realized_fraction_of_extent": float(np.mean(
                                [row["mean_realized_fraction_of_extent"] for row in cells])),
                            "mean_accuracy": float(np.mean([row["accuracy"] for row in cells])),
                            "mean_accuracy_drop_from_complete": float(np.mean(
                                [row["accuracy_drop_from_complete"] for row in cells])),
                            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in cells])),
                            "mean_mae": float(np.mean([row["mae"] for row in cells]))})
    write_csv(LANE / "bert_base_degradation_by_modality_rate.csv", grouped)
    return rows


def gaps(mask):
    runs = []
    start = None
    for i, missing in enumerate(mask.tolist() + [False]):
        if missing and start is None:
            start = i
        elif not missing and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def attachment3():
    with (RERUN / "bert_base_v5_attachment3_predictions.csv").open(
            newline="", encoding="utf-8-sig") as handle:
        predictions = {row["sample_id"]: row for row in csv.DictReader(handle)}
    paths = sorted(ATTACH.glob("*.pkl"))
    assert len(paths) == len(predictions) == 30
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            part = pickle.load(handle)["test"]
        assert "classification_labels" not in part and "regression_labels" not in part
        assert len(part["text_bert"]) == 1
        sample_id = f"{path.name}#0"
        extent = np.asarray(part["text_bert"][0, 1]) != 0
        token_ids = np.asarray(part["text_bert"][0, 0])
        content_extent = extent & ~np.isin(token_ids, (101, 102))
        observed = {
            "text": content_extent & (token_ids != 0),
            "audio": content_extent & np.any(np.asarray(part["audio"][0]) != 0, axis=1),
            "vision": content_extent & np.any(np.asarray(part["vision"][0]) != 0, axis=1),
        }
        row = {"sample_id": sample_id, "extent_slots": int(extent.sum()),
               "content_slots": int(content_extent.sum()),
               **predictions[sample_id]}
        row.pop("sample_id")
        row = {"sample_id": sample_id, **row}
        for modality, available in observed.items():
            missing = content_extent & ~available
            segments = gaps(missing)
            longest = max(segments, key=lambda item: item[1], default=(0, 0))
            row[f"{modality}_observed_slots"] = int(available.sum())
            row[f"{modality}_missing_slots"] = int(missing.sum())
            row[f"{modality}_missing_fraction"] = float(missing.sum() / max(1, content_extent.sum()))
            row[f"{modality}_gap_count"] = len(segments)
            row[f"{modality}_longest_gap_slots"] = int(longest[1])
            row[f"{modality}_longest_gap_start"] = int(longest[0]) if longest[1] else ""
        rows.append(row)
    write_csv(LANE / "attachment3_prediction_and_morphology.csv", rows)
    combinations = Counter()
    by_modality = {}
    for modality in ("text", "audio", "vision"):
        missing_rows = [row for row in rows if row[f"{modality}_missing_slots"] > 0]
        by_modality[modality] = {
            "samples_with_unobserved_slots": len(missing_rows),
            "mean_unobserved_fraction_all_samples": float(np.mean(
                [row[f"{modality}_missing_fraction"] for row in rows])),
            "mean_longest_gap_slots_when_present": float(np.mean(
                [row[f"{modality}_longest_gap_slots"] for row in missing_rows]))
                if missing_rows else 0.0,
        }
    for row in rows:
        combination = "+".join(modality for modality in ("text", "audio", "vision")
                               if row[f"{modality}_missing_slots"] > 0) or "none"
        combinations[combination] += 1
    morphology = {"n": len(rows), "mean_extent_slots": float(np.mean(
        [row["extent_slots"] for row in rows])),
        "mean_content_slots": float(np.mean([row["content_slots"] for row in rows])),
        "modality_combinations": dict(sorted(combinations.items())),
        "by_modality": by_modality,
        "interpretation": "Zero-valued content positions, excluding BERT CLS/SEP, are measured as unobserved; the features alone cannot distinguish imposed missingness from extraction failure or natural zeros, and have no physical timestamps."}
    (LANE / "attachment3_morphology_summary.json").write_text(
        json.dumps(morphology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def errors():
    data = np.load(OLD / "grid_student_bert_base_retest_seed1729.npz", allow_pickle=True)
    names = list(data["case_names"])
    assert names[0] == "complete" and len(names) == 113
    y = data["labels"].astype(int)
    truth = data["regression"].astype(float)
    pred = data["predicted_class"][0].astype(int)
    intensity = data["predicted_intensity"][0].astype(float)
    absolute = np.abs(intensity - truth)
    strict_conflict = ((pred == 0) & (intensity > 0)) | ((pred == 2) & (intensity < 0))
    quartile25, quartile75 = np.quantile(absolute, [0.25, 0.75])
    rows = []
    for i, sample_id in enumerate(data["sample_ids"]):
        rows.append({"sample_id": str(sample_id), "video_id": str(data["video_ids"][i]),
                     "true_class": int(y[i]), "pred_class": int(pred[i]),
                     "classification_correct": int(y[i] == pred[i]),
                     "error_type": "correct" if y[i] == pred[i] else f"{y[i]}_to_{pred[i]}",
                     "true_intensity": float(truth[i]),
                     "pred_intensity": float(intensity[i]),
                     "signed_regression_error": float(intensity[i] - truth[i]),
                     "absolute_regression_error": float(absolute[i]),
                     "opposite_polarity_conflict": int(strict_conflict[i]),
                     "large_regression_error": int(absolute[i] >= quartile75),
                     "small_regression_error": int(absolute[i] <= quartile25)})
    assert len(rows) == 728
    write_csv(LANE / "validation_error_attribution_per_sample.csv", rows)
    confusion = [[int(np.sum((y == a) & (pred == b))) for b in range(3)] for a in range(3)]
    error_types = dict(sorted(Counter(row["error_type"] for row in rows).items()))
    summary = {
        "source": "frozen_BERT_Base_FP16_package_validation_complete_case",
        "n": len(rows), "confusion_matrix": confusion,
        "classification_error_count": int(np.sum(y != pred)),
        "classification_error_types": error_types,
        "overall_mae": float(absolute.mean()),
        "mae_by_true_class": {str(k): float(absolute[y == k].mean()) for k in range(3)},
        "mae_when_classification_correct": float(absolute[y == pred].mean()),
        "mae_when_classification_wrong": float(absolute[y != pred].mean()),
        "regression_error_quartiles": {"q25": float(quartile25), "q75": float(quartile75)},
        "classification_correct_but_large_regression_error":
            int(np.sum((y == pred) & (absolute >= quartile75))),
        "classification_wrong_but_small_regression_error":
            int(np.sum((y != pred) & (absolute <= quartile25))),
        "opposite_polarity_conflict_count": int(strict_conflict.sum()),
        "opposite_polarity_conflict_definition":
            "Predicted negative with positive raw intensity, or predicted positive with negative raw intensity; neutral excluded",
    }
    (LANE / "validation_error_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    grid_rows = grid()
    attachment_rows = attachment3()
    error_summary = errors()
    print(json.dumps({"grid_cells": len(grid_rows), "attachment3_rows": len(attachment_rows),
                      "validation_samples": error_summary["n"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
