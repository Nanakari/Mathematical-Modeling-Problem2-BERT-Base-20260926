"""Verify the matched remote BERT-Base runs and summarize their metrics."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs/v5/student_bert_base_retest_seed1729.json"
FULL_OUTPUT = ROOT / "outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
NAMES = ("full", "text_only", "text_audio", "text_vision", "no_av_direct",
         "no_missing_aug", "full_seed2718", "full_seed3407")
EXPECTED = {
    "full": set(),
    "text_only": {"model.active_modalities"},
    "text_audio": {"model.active_modalities"},
    "text_vision": {"model.active_modalities"},
    "no_av_direct": {"model.av_direct"},
    "no_missing_aug": {"training.view_policy"},
    "full_seed2718": {"training.seed"},
    "full_seed3407": {"training.seed"},
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    base = read(BASE_CONFIG)
    reference = read(FULL_OUTPUT / "prepared_manifest.json")
    records = {}
    for name in NAMES:
        config_path = BASE_CONFIG if name == "full" else LANE / f"{name}_seed1729.json"
        if name in {"full_seed2718", "full_seed3407"}:
            config_path = LANE / f"{name}.json"
        output = FULL_OUTPUT if name == "full" else LANE / (
            name if name.startswith("full_seed") else f"{name}_seed1729")
        config = read(config_path)
        diffs = {f"{section}.{key}" for section in ("model", "data", "training", "distillation")
                 for key in set(base[section]) | set(config[section])
                 if base[section].get(key) != config[section].get(key)}
        assert diffs == EXPECTED[name], (name, diffs)
        prepared = read(output / "prepared_manifest.json")
        for key in ("train", "valid", "scaler_sha256", "source_hashes", "backbone_files"):
            assert prepared[key] == reference[key], (name, key)
        assert prepared["official_test_loaded_for_selection"] is False
        assert prepared["train_only_standardizer"] is True
        best = read(output / "best_validation.json")
        summary = read(output / "candidate_summary.json")
        history = [json.loads(line) for line in (output / "training_history.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert history[-1]["epoch"] == len(history)
        assert history[-1]["best_epoch"] == best["epoch"] == summary["best_epoch"]
        assert summary["checkpoint_sha256"] == sha(output / "best_model.safetensors")
        assert best["metrics"]["n"] == 728
        assert summary["official_test_evaluated"] is False
        records[name] = {"config": str(config_path), "output": str(output),
                         "changed_fields": sorted(diffs), "completed_epochs": len(history),
                         "best_epoch": best["epoch"], "metrics": best["metrics"],
                         "checkpoint_sha256": summary["checkpoint_sha256"]}
    metrics = {}
    for key in ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall"):
        values = [records[name]["metrics"][key] for name in
                  ("full", "full_seed2718", "full_seed3407")]
        metrics[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    result = {"same_data_scaler_source_and_backbone": True,
              "same_validation_selection_rule": True,
              "no_official_test_access": True,
              "runs": records, "three_seed_statistics": metrics}
    (LANE / "server_experiment_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"epochs": item["completed_epochs"],
                             "accuracy": item["metrics"]["accuracy"]}
                      for name, item in records.items()}))


if __name__ == "__main__":
    main()
