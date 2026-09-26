#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/a631/gaotianchang/bert_base_controlled_20260926
PY=/home/a631/gaotianchang/mm-p2-gpu/bin/python
LANE="$ROOT/outputs/bert_base_paper_20260926"
FEATURES="$ROOT/outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
cd "$ROOT"

for name in full text_only text_audio text_vision no_av_direct no_missing_aug; do
  if test "$name" = full; then
    config="$ROOT/configs/v5/student_bert_base_retest_seed1729.json"
    output="$ROOT/outputs/bert_base_vs_minilm_20260926/student_bert_base_retest_seed1729"
    script="$ROOT/v5_run.py"
  else
    config="$LANE/${name}_seed1729.json"
    output="$LANE/${name}_seed1729"
    script="$LANE/ablation_run.py"
  fi
  "$PY" "$script" evaluate --config "$config" \
    --checkpoint "$output/best_model.safetensors" --feature-path "$FEATURES" \
    --scenario-set quick_all_random --output "$output/validation_quick.json" \
    > "$LANE/${name}.server_quick_eval.log" 2>&1
  printf '%s completed\n' "$name" >> "$LANE/server_quick_eval_status.txt"
done
