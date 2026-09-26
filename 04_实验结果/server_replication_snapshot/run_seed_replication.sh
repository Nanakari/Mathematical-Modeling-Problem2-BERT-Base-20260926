#!/usr/bin/env bash
set -u
ROOT=/home/a631/gaotianchang/bert_base_controlled_20260926
PY=/home/a631/gaotianchang/mm-p2-gpu/bin/python
LANE="$ROOT/outputs/bert_base_paper_20260926"
FEATURES="$ROOT/outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
cd "$ROOT" || exit 1
for seed in 2718 3407; do
  for arm in text_only text_audio text_vision no_av_direct; do
    name="${arm}_seed${seed}"
    (
      set -e
      "$PY" "$LANE/ablation_run.py" prepare --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.prepare.log" 2>&1
      "$PY" "$LANE/ablation_run.py" train --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.train.log" 2>&1
    ) > "$LANE/$name.runner.log" 2>&1 &
  done
  wait
  for arm in text_only text_audio text_vision no_av_direct; do
    name="${arm}_seed${seed}"
    if test -f "$LANE/$name/candidate_summary.json"; then echo "$name complete"; else echo "$name failed"; fi
  done
done
