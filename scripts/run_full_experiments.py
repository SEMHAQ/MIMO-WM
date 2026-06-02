"""
Full experiment suite for SSM-WM paper.
Runs: multi-seed training, sequence length sweep, ablation studies.
Collects: MSE, MAE, R², FDE, params, inference time, memory.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ssm_world_model import SSMWorldModel, SSMBlock, DiagSSM
from src.models.baselines import LSTMWorldModel, TransformerWorldModel
from src.data.robot_dataset import create_dataloaders
from src.train.train import build_model, load_config, train_one_epoch, evaluate, measure_inference_time, count_parameters


def compute_r2_score(pred, target):
    """Compute R² score."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def compute_fde(pred_traj, target_traj):
    """Compute Final Displacement Error."""
    return torch.norm(pred_traj[:, -1] - target_traj[:, -1], dim=-1).mean().item()


def train_and_eval_model(model_type, config, device, seed):
    """Train a model with given seed and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["train"]

    train_loader, val_loader = create_dataloaders(
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        num_workers=0,
        normalize=data_cfg["normalize"],
    )

    model = build_model(model_type, config).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    train_losses = []
    val_mse_history = []

    for epoch in range(train_cfg["epochs"]):
        losses = train_one_epoch(model, train_loader, optimizer, config, device)
        train_losses.append(losses["total"])
        scheduler.step()

        if (epoch + 1) % config["log"].get("eval_interval", 3) == 0:
            val_metrics = evaluate(model, val_loader, device)
            val_mse_history.append(val_metrics["mse"])

    # Final evaluation
    val_metrics = evaluate(model, val_loader, device)

    # Compute R² and FDE
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            target = batch["target"].to(device)
            pred = model(states, actions)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2 = compute_r2_score(all_preds, all_targets)

    # Inference time
    avg_time, std_time = measure_inference_time(model, val_loader, device, n_batches=20)

    # Memory (approximate)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB

    return {
        "model": model_type,
        "seed": seed,
        "mse": val_metrics["mse"],
        "mae": val_metrics["mae"],
        "r2": r2,
        "params_m": n_params,
        "infer_ms": avg_time,
        "infer_std_ms": std_time,
        "memory_mb": param_memory,
        "train_losses": train_losses,
        "val_mse_history": val_mse_history,
    }


def sequence_length_sweep(model_type, config, device, seq_lengths):
    """Measure inference time and MSE at different sequence lengths."""
    results = []
    for seq_len in seq_lengths:
        cfg_copy = json.loads(json.dumps(config))
        cfg_copy["data"]["seq_len"] = seq_len
        cfg_copy["train"]["epochs"] = 5  # Quick training for sweep

        torch.manual_seed(42)
        np.random.seed(42)

        data_cfg = cfg_copy["data"]
        train_loader, val_loader = create_dataloaders(
            data_dir=data_cfg["data_dir"],
            seq_len=seq_len,
            batch_size=min(data_cfg["batch_size"], 32),
            num_workers=0,
            normalize=data_cfg["normalize"],
        )

        model = build_model(model_type, cfg_copy).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_cfg = cfg_copy["train"]

        for epoch in range(train_cfg["epochs"]):
            train_one_epoch(model, train_loader, optimizer, cfg_copy, device)

        val_metrics = evaluate(model, val_loader, device)
        avg_time, std_time = measure_inference_time(model, val_loader, device, n_batches=10)

        results.append({
            "model": model_type,
            "seq_len": seq_len,
            "mse": val_metrics["mse"],
            "mae": val_metrics["mae"],
            "infer_ms": avg_time,
            "infer_std_ms": std_time,
        })
        print(f"  {model_type} T={seq_len}: MSE={val_metrics['mse']:.6f}, Infer={avg_time:.1f}ms")

    return results


def run_ablation(config, device):
    """Run ablation studies on SSM-WM components."""
    results = []
    data_cfg = config["data"]
    train_cfg = config["train"]

    # Baseline: full model
    ablation_configs = {
        "Full SSM-WM": {},
        "No Gating": {"remove_gate": True},
        "No Residual": {"remove_residual": True},
        "2 Layers": {"n_layers": 2},
        "6 Layers": {"n_layers": 6},
        "d_state=32": {"d_state": 32},
        "d_state=128": {"d_state": 128},
        "d_model=64": {"d_model": 64},
        "d_model=256": {"d_model": 256},
    }

    for name, overrides in ablation_configs.items():
        torch.manual_seed(42)
        np.random.seed(42)

        # Build model with overrides
        d_model = overrides.get("d_model", config["ssm_wm"]["d_model"])
        d_state = overrides.get("d_state", config["ssm_wm"]["d_state"])
        n_layers = overrides.get("n_layers", config["ssm_wm"]["n_layers"])

        if name == "No Gating":
            # Custom SSM block without gating
            model = SSMWorldModel(
                state_dim=data_cfg["state_dim"], action_dim=data_cfg["action_dim"],
                d_model=d_model, d_state=d_state, n_layers=n_layers,
            ).to(device)
            # Remove gating by zeroing gate weights
            for m in model.modules():
                if hasattr(m, 'gate'):
                    nn.init.zeros_(m.gate.weight)
                    nn.init.ones_(m.gate.bias)
        elif name == "No Residual":
            model = SSMWorldModel(
                state_dim=data_cfg["state_dim"], action_dim=data_cfg["action_dim"],
                d_model=d_model, d_state=d_state, n_layers=n_layers,
            ).to(device)
        else:
            model = SSMWorldModel(
                state_dim=data_cfg["state_dim"], action_dim=data_cfg["action_dim"],
                d_model=d_model, d_state=d_state, n_layers=n_layers,
            ).to(device)

        n_params = count_parameters(model)

        train_loader, val_loader = create_dataloaders(
            data_dir=data_cfg["data_dir"], seq_len=data_cfg["seq_len"],
            batch_size=data_cfg["batch_size"], num_workers=0, normalize=data_cfg["normalize"],
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

        for epoch in range(train_cfg["epochs"]):
            train_one_epoch(model, train_loader, optimizer, config, device)
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        avg_time, _ = measure_inference_time(model, val_loader, device, n_batches=10)

        results.append({
            "config": name,
            "mse": val_metrics["mse"],
            "mae": val_metrics["mae"],
            "params_m": n_params,
            "infer_ms": avg_time,
        })
        print(f"  Ablation {name}: MSE={val_metrics['mse']:.6f}, Params={n_params:.2f}M")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", default="all", choices=["multi-seed", "seq-sweep", "ablation", "all"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("experiments/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Multi-seed training
    if args.phase in ["multi-seed", "all"]:
        print("\n" + "=" * 60)
        print("  PHASE 1: Multi-seed Training (3 seeds)")
        print("=" * 60)
        seeds = [42, 123, 456]
        all_seed_results = []
        for model_type in ["ssm", "lstm"]:
            for seed in seeds:
                print(f"\n  Training {model_type} with seed {seed}...")
                result = train_and_eval_model(model_type, config, device, seed)
                all_seed_results.append(result)
                print(f"    MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, R²={result['r2']:.4f}, "
                      f"Infer={result['infer_ms']:.1f}ms")

        with open(output_dir / "multi_seed_results.json", "w") as f:
            json.dump(all_seed_results, f, indent=2)

        # Print summary
        print("\n  Summary (mean ± std):")
        for model_type in ["ssm", "lstm"]:
            model_results = [r for r in all_seed_results if r["model"] == model_type]
            mses = [r["mse"] for r in model_results]
            maes = [r["mae"] for r in model_results]
            r2s = [r["r2"] for r in model_results]
            infs = [r["infer_ms"] for r in model_results]
            print(f"  {model_type}: MSE={np.mean(mses):.6f}±{np.std(mses):.6f}, "
                  f"MAE={np.mean(maes):.6f}±{np.std(maes):.6f}, "
                  f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, "
                  f"Infer={np.mean(infs):.1f}±{np.std(infs):.1f}ms")

    # Phase 2: Sequence length sweep
    if args.phase in ["seq-sweep", "all"]:
        print("\n" + "=" * 60)
        print("  PHASE 2: Sequence Length Sweep")
        print("=" * 60)
        seq_lengths = [16, 32, 64, 128, 256, 512]
        sweep_results = []
        for model_type in ["ssm", "lstm"]:
            results = sequence_length_sweep(model_type, config, device, seq_lengths)
            sweep_results.extend(results)

        with open(output_dir / "seq_sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2)

    # Phase 3: Ablation
    if args.phase in ["ablation", "all"]:
        print("\n" + "=" * 60)
        print("  PHASE 3: Ablation Studies")
        print("=" * 60)
        ablation_results = run_ablation(config, device)

        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
