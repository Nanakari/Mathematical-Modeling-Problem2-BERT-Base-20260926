#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/a631/gaotianchang/bert_base_controlled_20260926
PY=/home/a631/gaotianchang/mm-p2-gpu/bin/python
LANE="$ROOT/outputs/bert_base_paper_20260926"
FEATURES="$ROOT/outputs/teacher_student_v5_20260924/data/train_valid_only_v5.pkl"
cd "$ROOT"
while ! test -f "$LANE/no_av_direct_seed3407/candidate_summary.json"; do sleep 10; done
for seed in 2718 3407; do
 for arm in full no_av_direct; do
  name="${arm}_seed${seed}"
  "$PY" "$LANE/ablation_run.py" evaluate --config "$LANE/$name.json" --checkpoint "$LANE/$name/best_model.safetensors" --feature-path "$FEATURES" --scenario-set quick_all_random --output "$LANE/$name/validation_quick.json" > "$LANE/$name.quick.log" 2>&1
  echo "$name quick complete"
 done
done
"$PY" "$LANE/evaluate_three_seed_test.py" > "$LANE/three_seed_test.log" 2>&1
echo 'three seed test complete'
