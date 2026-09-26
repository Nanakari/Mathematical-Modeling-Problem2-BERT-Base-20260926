#!/usr/bin/env bash
set -u
ROOT=/home/a631/gaotianchang/bert_base_controlled_20260926
PY=/home/a631/gaotianchang/mm-p2-gpu/bin/python
LANE="$ROOT/outputs/bert_base_paper_20260926"
FEATURES="$ROOT/outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$ROOT" || exit 1

for name in full_seed2718 full_seed3407 no_missing_aug_seed1729; do
  if test "$name" = no_missing_aug_seed1729; then
    script="$LANE/ablation_run.py"
  else
    script="$ROOT/v5_run.py"
  fi
  (
    set -e
    "$PY" "$script" prepare --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.server_prepare.log" 2>&1
    "$PY" "$script" train --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.server_train.log" 2>&1
  ) > "$LANE/$name.server_runner.log" 2>&1 &
  printf '%s %s\n' "$name" "$!" >> "$LANE/server_additional_pids.txt"
done

wait
for name in full_seed2718 full_seed3407 no_missing_aug_seed1729; do
  if test -f "$LANE/$name/candidate_summary.json"; then
    printf '%s completed\n' "$name" >> "$LANE/server_additional_status.txt"
  else
    printf '%s failed_or_incomplete\n' "$name" >> "$LANE/server_additional_status.txt"
  fi
done
