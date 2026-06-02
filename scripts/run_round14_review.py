"""
Round 14 peer review simulation.
Reads the paper and generates 5 reviewer reports.
"""
import os
import sys
import json
from pathlib import Path

# Read the paper
paper_path = Path("paper/main.tex")
with open(paper_path, "r", encoding="utf-8") as f:
    paper_content = f.read()

# Read experiment results
results_dir = Path("experiments/paper_results")
results = {}

for f in results_dir.glob("*.json"):
    with open(f, "r") as fh:
        results[f.stem] = json.load(fh)

# Print summary of what we have
print("="*60)
print("  Round 14 Review Preparation")
print("="*60)
print(f"\nPaper length: {len(paper_content)} characters")
print(f"\nAvailable result files:")
for k, v in results.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} entries")
    else:
        print(f"  {k}: {type(v).__name__}")

# Check for key results
if "multi_seed_results_5seeds" in results:
    data = results["multi_seed_results_5seeds"]
    print("\n  5-seed results available:")
    for model in ["ssm", "lstm"]:
        model_data = [r for r in data if r["model"] == model]
        if model_data:
            mses = [r["mse"] for r in model_data]
            r2s = [r["r2"] for r in model_data]
            infs = [r["infer_ms"] for r in model_data]
            import numpy as np
            print(f"    {model}: MSE={np.mean(mses):.6f}±{np.std(mses):.6f}, "
                  f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, "
                  f"Infer={np.mean(infs):.1f}±{np.std(infs):.1f}ms")

if "mpc_results" in results:
    data = results["mpc_results"]
    print("\n  MPC results available:")
    for model in ["ssm", "lstm"]:
        model_data = [r for r in data if r["model"] == model]
        if model_data:
            errors = [r["mean_tracking_error"] for r in model_data]
            times = [r["mean_control_time_ms"] for r in model_data]
            import numpy as np
            print(f"    {model}: Tracking Error={np.mean(errors):.4f}±{np.std(errors):.4f}, "
                  f"Control Time={np.mean(times):.1f}ms")

print("\nReady for Round 14 review!")
