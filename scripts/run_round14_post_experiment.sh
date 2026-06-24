#!/bin/bash
# Post-experiment workflow for Round 14
# Run this after the main experiment (PID 16651) completes

set -e

echo "=============================================="
echo "  Round 14 Post-Experiment Workflow"
echo "=============================================="

# Step 1: Update paper tables with 5-seed results
echo ""
echo "Step 1: Updating paper tables..."
python3 scripts/update_paper_tables.py

# Step 2: Run MuJoCo MPC experiment
echo ""
echo "Step 2: Running MuJoCo MPC experiment..."
python3 scripts/run_round14_experiments.py --config configs/mujoco.yaml --experiment mpc

# Step 3: Run Round 14 peer review simulation
echo ""
echo "Step 3: Running Round 14 peer review simulation..."
python3 scripts/run_round14_review.py

# Step 4: Check scores
echo ""
echo "Step 4: Checking review scores..."
SCORES_FILE="experiments/paper_results/round14_scores.json"
if [ -f "$SCORES_FILE" ]; then
    python3 -c "
import json
with open('$SCORES_FILE') as f:
    data = json.load(f)
scores = data['reviewer_scores']
grand_mean = data['grand_mean']
all_pass = data['all_pass_85']
print(f'Grand Mean: {grand_mean:.2f}')
print(f'All >= 85: {all_pass}')
for r in ['eic', 'r1', 'r2', 'r3', 'da']:
    s = scores[r]['overall']
    print(f'  {r.upper()}: {s} {\"✅\" if s >= 85 else \"❌\"}')
"
fi

echo ""
echo "=============================================="
echo "  Workflow Complete!"
echo "=============================================="
