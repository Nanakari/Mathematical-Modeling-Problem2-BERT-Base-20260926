"""Export paired, validation-only per-sample predictions for selected arms."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LANE))

import ablation_run  # noqa: F401; registers input masking for controlled arms
import v5_train
from v5_data import load_tokenizers, load_train_valid

FEATURES = ROOT / "outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
BASE = ROOT / "outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
CONFIG = ROOT / "configs/v5/student_bert_base_retest_seed1729.json"
ARMS = (
    ("full", CONFIG, BASE),
    ("text_only", LANE / "text_only_seed1729.json", LANE / "text_only_seed1729"),
    ("text_audio", LANE / "text_audio_seed1729.json", LANE / "text_audio_seed1729"),
    ("text_vision", LANE / "text_vision_seed1729.json", LANE / "text_vision_seed1729"),
    ("no_av_direct", LANE / "no_av_direct_seed1729.json", LANE / "no_av_direct_seed1729"),
    ("no_missing_aug", LANE / "no_missing_aug_seed1729.json", LANE / "no_missing_aug_seed1729"),
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=[item[0] for item in ARMS],
                        default=[item[0] for item in ARMS])
    parser.add_argument("--output-prefix", default="controlled_valid")
    args = parser.parse_args(argv)
    _, valid = load_train_valid(FEATURES)
    samples = [valid[index] for index in range(len(valid))]
    part = valid.part
    y = np.asarray(part["classification_labels"], dtype=int)
    reg = np.asarray(part["regression_labels"], dtype=float)
    ids = [str(item) for item in part["id"]]
    assert len(ids) == len(set(ids)) == len(y) == 728
    rows = [{"sample_id": ids[i], "true_class": int(y[i]), "true_intensity": float(reg[i])}
            for i in range(len(y))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    for name, config_path, output in ARMS:
        if name not in args.arms:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = output / "best_model.safetensors"
        model = v5_train._load_best_model(config, checkpoint, device=device,
                                          with_distill_heads=False)
        normalizer = v5_train._load_standardizer(output / "scaler.json")
        backbone = v5_train.resolve_project_path(config["model"]["backbone_dir"])
        source_tok, target_tok = load_tokenizers(backbone,
                                                  source_vocab=v5_train.source_vocab_from(config))
        metrics, logits, intensity = v5_train._evaluate(
            model, samples, y, reg, normalizer, source_tok, target_tok,
            device=device, batch_size=32,
            max_text_length=int(config["data"]["max_text_length"]),
            amp_dtype=config["training"]["amp_dtype"])
        best = json.loads((output / "best_validation.json").read_text(encoding="utf-8"))
        for key in ("accuracy", "macro_f1", "mae", "pearson"):
            if abs(metrics[key] - best["metrics"][key]) > 1e-6:
                raise AssertionError((name, key, metrics[key], best["metrics"][key]))
        scores = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1)
        for i, row in enumerate(rows):
            row[f"{name}_pred_class"] = int(pred[i])
            row[f"{name}_prob_negative"] = float(probs[i, 0])
            row[f"{name}_prob_neutral"] = float(probs[i, 1])
            row[f"{name}_prob_positive"] = float(probs[i, 2])
            row[f"{name}_pred_intensity"] = float(intensity[i])
            row[f"{name}_correct"] = int(pred[i] == y[i])
            row[f"{name}_absolute_error"] = float(abs(intensity[i] - reg[i]))
        summary[name] = {"best_epoch": best["epoch"], "metrics": metrics,
                         "checkpoint": str(checkpoint)}
        print(name, metrics["accuracy"], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (LANE / f"{args.output_prefix}_per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (LANE / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
