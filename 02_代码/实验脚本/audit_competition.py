"""Audit question-2 experimental protocol against the local competition data."""
import csv
import hashlib
import json
import pickle
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
FEATURES = ROOT / "outputs" / "teacher_student_v5_20260924" / "data" / "train_valid_only_v5.pkl"
ORIGINAL = ROOT / "outputs" / "bert_base_vs_minilm_20260926"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    with FEATURES.open("rb") as f:
        data = pickle.load(f)
    assert set(data) == {"train", "valid"}
    counts = {s: len(data[s]["text_bert"]) for s in data}
    assert counts == {"train": 3395, "valid": 728}
    labels = {}
    videos = {}
    for split in ("train", "valid"):
        part = data[split]
        cls = np.asarray(part["classification_labels"]).astype(int)
        reg = np.asarray(part["regression_labels"]).astype(float)
        assert np.array_equal(cls, np.sign(reg).astype(int) + 1)
        labels[split] = np.bincount(cls, minlength=3).tolist()
        ids = [str(x) for x in part["id"]]
        assert len(set(ids)) == len(ids)
        videos[split] = {sid.split("$_$")[0] for sid in ids}
        assert np.shape(part["text_bert"]) == (counts[split], 3, 50)
        assert np.shape(part["audio"]) == (counts[split], 50, 74)
        assert np.shape(part["vision"]) == (counts[split], 50, 35)
    assert not videos["train"].intersection(videos["valid"])
    base = json.loads((ROOT / "configs/v5/student_bert_base_retest_seed1729.json").read_text(encoding="utf-8"))
    config_diffs = {}
    for config_path in sorted(LANE.glob("*seed*.json")):
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        diffs = {}
        for section in ("model", "data", "training", "distillation"):
            for key in set(base[section]) | set(cfg[section]):
                old, new = base[section].get(key), cfg[section].get(key)
                if old != new:
                    diffs[f"{section}.{key}"] = {"base": old, "experiment": new}
        config_diffs[config_path.stem] = diffs
    expected = {
        "full_seed2718": {"training.seed"}, "full_seed3407": {"training.seed"},
        "no_missing_aug_seed1729": {"training.view_policy"},
    }
    assert set(config_diffs) == set(expected)
    for name in expected:
        assert set(config_diffs[name]) == expected[name], (name, config_diffs[name])
    reference_prep = json.loads((ORIGINAL / "student_bert_base_retest_seed1729" /
                                 "prepared_manifest.json").read_text(encoding="utf-8"))
    prepared_runs = {}
    for name in expected:
        prep_path = LANE / name / "prepared_manifest.json"
        if prep_path.exists():
            prep = json.loads(prep_path.read_text(encoding="utf-8"))
            for key in ("train", "valid", "scaler_sha256", "source_hashes", "backbone_files"):
                assert prep[key] == reference_prep[key], (name, key)
            assert prep["official_test_loaded_for_selection"] is False
            assert prep["train_only_standardizer"] is True
            prepared_runs[name] = True
    freeze = json.loads((ORIGINAL / "freeze_student_bert_base_retest_seed1729.json").read_text(encoding="utf-8"))
    test = json.loads((ORIGINAL / "student_bert_base_retest_seed1729_official_test.json").read_text(encoding="utf-8"))
    assert freeze["official_test_evaluated"] and test["status"] == "evaluated_once_after_freeze"
    assert test["official_test"]["n"] == 727
    assert not list(LANE.glob("*official_test*.json"))
    attachment_path = ROOT / "outputs/two_model_rerun_20260926/bert_base_v5_attachment3_predictions.csv"
    with attachment_path.open(encoding="utf-8-sig", newline="") as f:
        attachment_rows = list(csv.DictReader(f))
    assert len(attachment_rows) == 30
    assert all({"sample_id", "predicted_class", "intensity_raw"} <= set(row)
               for row in attachment_rows)
    attachment_dir = ROOT.parent / "E题" / "E题数据" / "附件3-模态缺失特征样本" / "对齐版本"
    attachment_files = sorted(attachment_dir.glob("*.pkl"))
    assert len(attachment_files) == 30
    for path in attachment_files:
        with path.open("rb") as f:
            raw = pickle.load(f)["test"]
        assert "classification_labels" not in raw and "regression_labels" not in raw
    old = ORIGINAL / "student_bert_base_retest_seed1729"
    new = LANE / "full_seed2718"
    original_best = json.loads((old / "best_validation.json").read_text(encoding="utf-8"))
    new_best = json.loads((new / "best_validation.json").read_text(encoding="utf-8"))
    reevaluated = json.loads((new / "validation_quick.json").read_text(encoding="utf-8"))
    assert new_best["metrics"] == reevaluated["complete"]
    history = [json.loads(line) for line in (new / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(history) == 6 and new_best["epoch"] == 5
    assert history[-1]["best_epoch"] == new_best["epoch"]
    metrics = {}
    for key in ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall"):
        values = [original_best["metrics"][key], new_best["metrics"][key]]
        metrics[key] = {"values": dict(zip(("1729", "2718"), values)),
                        "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    grid = json.loads((ORIGINAL / "grid_student_bert_base_retest_seed1729.json").read_text(encoding="utf-8"))
    assert grid["protocol"]["cell_count"] == 113
    assert grid["protocol"]["validation_samples"] == 728
    assert grid["protocol"]["official_test_evaluated"] is False
    with (LANE / "realized_missing_lengths.csv").open(encoding="utf-8-sig", newline="") as f:
        lengths = list(csv.DictReader(f))
    realized_rates = {str(rate): statistics.mean(float(row["mean_realized_fraction_of_extent"])
                     for row in lengths if float(row["nominal_rate"]) == rate)
                     for rate in (0.1, 0.3, 0.5, 0.7)}
    package_zip = ORIGINAL / "package_student_bert_base_retest_seed1729.zip"
    findings = {
        "training_source": {"path": str(FEATURES), "sha256": sha(FEATURES),
                            "keys": sorted(data), "counts": counts, "label_counts": labels,
                            "cross_split_video_overlap": 0},
        "model": {"backbone": base["model_id"], "revision": base["model_revision"],
                  "source": "fixed google-bert/bert-base-uncased general pretraining; see pretrained_provenance.json"},
        "controlled_config_changes": config_diffs,
        "prepared_runs_matching_primary_data_scaler_source_and_backbone": prepared_runs,
        "seed_replication": {"seed_1729": {"status": "completed", "best_epoch": original_best["epoch"]},
                             "seed_2718": {"status": "user_stopped_after_epoch_6", "best_epoch": new_best["epoch"],
                                           "checkpoint_sha256": sha(new / "best_model.safetensors"),
                                           "best_checkpoint_reevaluation_identical": True},
                             "seed_3407": {"status": "not_started"},
                             "two_seed_validation_metrics": metrics,
                             "limits": "n=2; seed 2718 was user-stopped, so training budgets differ"},
        "ablation_training": "no_missing_aug_seed1729 configured but not started at user direction",
        "ablation_wrapper_sha256": sha(LANE / "ablation_run.py"),
        "historical_test": {"n": test["official_test"]["n"],
                            "accuracy": test["official_test"]["accuracy"],
                            "frozen_package_sha256": test["package_sha256"],
                            "previously_accessed_in_project": True,
                            "no_new_test_files_in_this_lane": True},
        "attachment3_rows": len(attachment_rows),
        "attachment3_feature_files": len(attachment_files),
        "attachment3_has_no_labels": True,
        "existing_missing_scenario_grid": {"cells": 113, "missing_cells": 112,
                                            "source": "validation_only", "model_seed": 1729},
        "realized_fraction_of_aligned_extent_by_nominal_rate": realized_rates,
        "missing_duration_measurement": "realized_missing_lengths.csv measures removed observed aligned slots; source files contain no physical seconds per position",
        "package_zip_bytes": package_zip.stat().st_size,
        "scope_exception": "50 MB submission attachment limit explicitly deferred by user; BERT-Base package remains oversized",
    }
    out = LANE / "competition_audit.json"
    out.write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    run()
