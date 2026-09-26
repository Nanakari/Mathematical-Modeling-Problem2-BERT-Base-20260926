"""Matched validation input occlusion on each trained full BERT-Base seed.

This is a diagnostic of dependence at inference, distinct from retraining an
ablation arm. It never reads test or attachment 3.
"""
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LANE))

import ablation_run  # patches V5Model.forward for modality occlusion
import v5_train
from v5_data import load_tokenizers, load_train_valid

FEATURES = ROOT / "outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
ORIGINAL = ROOT / "outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"


def main():
    _, valid = load_train_valid(FEATURES)
    samples = [valid[i] for i in range(len(valid))]
    y = valid.part["classification_labels"].astype(int)
    r = valid.part["regression_labels"].astype(float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed, output, config_path in (
        (1729, ORIGINAL, ROOT / "configs/v5/student_bert_base_retest_seed1729.json"),
        (2718, LANE / "full_seed2718", LANE / "full_seed2718.json"),
        (3407, LANE / "full_seed3407", LANE / "full_seed3407.json"),
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = v5_train._load_best_model(config, output / "best_model.safetensors",
                                          device=device, with_distill_heads=False)
        normalizer = v5_train._load_standardizer(output / "scaler.json")
        backbone = v5_train.resolve_project_path(config["model"]["backbone_dir"])
        source_tok, target_tok = load_tokenizers(backbone,
                                                  source_vocab=v5_train.source_vocab_from(config))
        for name, active in (("full", None), ("text_only", ["text"]),
                             ("text_audio", ["text", "audio"]),
                             ("text_vision", ["text", "vision"])):
            if active is None:
                model.model_config.pop("active_modalities", None)
            else:
                model.model_config["active_modalities"] = active
            metrics, _, _ = v5_train._evaluate(
                model, samples, y, r, normalizer, source_tok, target_tok,
                device=device, batch_size=32, max_text_length=128, amp_dtype="fp16")
            rows.append({"seed": seed, "occlusion": name, **{k: metrics[k] for k in
                         ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall")}})
            print(seed, name, metrics["accuracy"], flush=True)
        model.model_config.pop("active_modalities", None)
        old_scale = model.av_direct_scale
        model.av_direct_scale = 0.0
        metrics, _, _ = v5_train._evaluate(
            model, samples, y, r, normalizer, source_tok, target_tok,
            device=device, batch_size=32, max_text_length=128, amp_dtype="fp16")
        model.av_direct_scale = old_scale
        rows.append({"seed": seed, "occlusion": "no_av_direct", **{k: metrics[k] for k in
                     ("accuracy", "macro_f1", "mae", "pearson", "neutral_recall")}})
        print(seed, "no_av_direct", metrics["accuracy"], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (LANE / "inference_occlusion_valid.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
