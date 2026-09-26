#!/usr/bin/env bash
set -u
ROOT=/home/a631/gaotianchang/bert_base_controlled_20260926
PY=/home/a631/gaotianchang/mm-p2-gpu/bin/python
LANE="$ROOT/outputs/bert_base_paper_20260926"
FEATURES="$ROOT/outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
SCRIPT="$LANE/ablation_run.py"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$ROOT" || exit 1

for name in text_only_seed1729 text_audio_seed1729 text_vision_seed1729 no_av_direct_seed1729; do
  (
    set -e
    "$PY" "$SCRIPT" prepare --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.prepare.log" 2>&1
    "$PY" "$SCRIPT" train --config "$LANE/$name.json" --feature-path "$FEATURES" > "$LANE/$name.train.log" 2>&1
  ) > "$LANE/$name.runner.log" 2>&1 &
  pid=$!
  printf '%s %s\n' "$name" "$pid" >> "$LANE/server_pids.txt"
done

wait
for name in text_only_seed1729 text_audio_seed1729 text_vision_seed1729 no_av_direct_seed1729; do
  if test -f "$LANE/$name/candidate_summary.json"; then
    printf '%s completed\n' "$name" >> "$LANE/server_status.txt"
  else
    printf '%s failed_or_incomplete\n' "$name" >> "$LANE/server_status.txt"
  fi
done
