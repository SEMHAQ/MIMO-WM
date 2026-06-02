"""
Update paper tables with 5-seed experiment results.
Reads multi_seed_results_5seeds.json and mpc_results.json,
then updates main.tex tables with mean ± std values.
"""
import json
import numpy as np
from pathlib import Path


def load_results():
    results_dir = Path("experiments/paper_results")
    results = {}
    for f in results_dir.glob("*.json"):
        with open(f, "r") as fh:
            results[f.stem] = json.load(fh)
    return results


def compute_stats(data, model_type, metric):
    model_data = [r for r in data if r["model"] == model_type]
    values = [r[metric] for r in model_data]
    return np.mean(values), np.std(values)


def format_val(mean, std, scale=1, fmt=".2f"):
    """Format mean ± std with scaling."""
    return f"{mean*scale:{fmt}}$\\pm${std*scale:{fmt}}"


def main():
    results = load_results()

    # Load 5-seed results
    if "multi_seed_results_5seeds" in results:
        data = results["multi_seed_results_5seeds"]
    elif "multi_seed_results" in results:
        data = results["multi_seed_results"]
    else:
        print("ERROR: No multi-seed results found!")
        return

    print("="*60)
    print("  Paper Table Update")
    print("="*60)

    # Compute stats for each model
    models = ["ssm", "lstm"]
    stats = {}

    for model in models:
        model_data = [r for r in data if r["model"] == model]
        if not model_data:
            print(f"  WARNING: No data for {model}")
            continue

        stats[model] = {}
        for metric in ["mse", "mae", "r2", "infer_ms", "params_m"]:
            values = [r[metric] for r in model_data]
            stats[model][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

        # Multi-step MSE
        if "multi_step_mse" in model_data[0]:
            for h_key in ["h4", "h8", "h16"]:
                values = [r["multi_step_mse"].get(h_key, 0) for r in model_data]
                stats[model][f"multi_step_{h_key}"] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

    # Print Table 1: Main Results
    print("\n--- Table 1: Main Results (T=64) ---")
    print(f"{'Method':<25} {'MSE(×10⁻³)':<20} {'MAE':<20} {'R²':<20} {'Params/M':<10}")
    print("-" * 95)

    for model in ["lstm", "ssm"]:
        if model not in stats:
            continue
        s = stats[model]
        mse_str = format_val(s["mse"]["mean"], s["mse"]["std"], 1000, ".2f")
        mae_str = format_val(s["mae"]["mean"], s["mae"]["std"], 1, ".3f")
        r2_str = format_val(s["r2"]["mean"], s["r2"]["std"], 1, ".3f")
        params_str = f"{s['params_m']['mean']:.2f}"
        label = "SSM-WM(本文)" if model == "ssm" else "LSTM-WM(2层)"
        print(f"{label:<25} {mse_str:<20} {mae_str:<20} {r2_str:<20} {params_str:<10}")

    # Print Table 2: Inference Performance
    print("\n--- Table 2: Inference Performance (T=64, batch 64) ---")
    print(f"{'Method':<25} {'Infer/ms':<20} {'Speedup':<10}")
    print("-" * 55)

    if "lstm" in stats:
        lstm_infer = stats["lstm"]["infer_ms"]["mean"]
        for model in ["lstm", "ssm"]:
            if model not in stats:
                continue
            s = stats[model]
            infer_str = format_val(s["infer_ms"]["mean"], s["infer_ms"]["std"], 1, ".1f")
            speedup = lstm_infer / s["infer_ms"]["mean"]
            label = "SSM-WM(本文)" if model == "ssm" else "LSTM-WM"
            print(f"{label:<25} {infer_str:<20} {speedup:.1f}×")

    # Print MPC results
    if "mpc_results" in results:
        mpc_data = results["mpc_results"]
        print("\n--- Table 7: MPC Control Performance ---")
        print(f"{'Method':<25} {'Tracking MSE':<20} {'Control Time/ms':<20} {'Freq/Hz':<10}")
        print("-" * 75)

        for model in ["lstm", "ssm"]:
            model_data = [r for r in mpc_data if r["model"] == model]
            if not model_data:
                continue

            errors = [r["mean_tracking_error"] for r in model_data]
            times = [r["mean_control_time_ms"] for r in model_data]

            err_str = f"{np.mean(errors):.4f}$\\pm${np.std(errors):.4f}"
            time_str = f"{np.mean(times):.0f}$\\pm${np.std(times):.0f}"
            freq = 1000.0 / np.mean(times)
            freq_str = f"{freq:.1f}"

            label = "SSM-WM-MPC" if model == "ssm" else "LSTM-MPC"
            print(f"{label:<25} {err_str:<20} {time_str:<20} {freq_str:<10}")

    # Generate LaTeX table updates
    print("\n" + "="*60)
    print("  LaTeX Table Updates")
    print("="*60)

    if "ssm" in stats and "lstm" in stats:
        ssm = stats["ssm"]
        lstm = stats["lstm"]

        # Table 1 values
        print("\n% Table 1 updates:")
        print(f"% SSM-WM: MSE={ssm['mse']['mean']*1000:.2f}±{ssm['mse']['std']*1000:.2f}, "
              f"MAE={ssm['mae']['mean']:.3f}±{ssm['mae']['std']:.3f}, "
              f"R²={ssm['r2']['mean']:.3f}±{ssm['r2']['std']:.3f}")
        print(f"% LSTM-WM: MSE={lstm['mse']['mean']*1000:.2f}±{lstm['mse']['std']*1000:.2f}, "
              f"MAE={lstm['mae']['mean']:.3f}±{lstm['mae']['std']:.3f}, "
              f"R²={lstm['r2']['mean']:.3f}±{lstm['r2']['std']:.3f}")

    # Save summary JSON
    summary = {}
    for model in stats:
        summary[model] = {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                         for k, v in stats[model].items()}

    summary_path = Path("experiments/paper_results/table_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {summary_path}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
