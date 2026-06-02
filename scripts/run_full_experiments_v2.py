"""
Full experiment suite v2 for SSM-WM paper.
Handles cuDNN errors by falling back to CPU for LSTM.
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

from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel
from src.data.robot_dataset import create_dataloaders
from src.train.train import build_model, load_config, train_one_epoch, evaluate, measure_inference_time, count_parameters


def compute_r2_score(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def train_and_eval_model(model_type, config, device, seed):
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

    # Build model - try GPU first, fall back to CPU for LSTM if cuDNN fails
    try:
        model = build_model(model_type, config).to(device)
    except RuntimeError as e:
        if "cuDNN" in str(e) or "cudnn" in str(e):
            print(f"    cuDNN error, falling back to CPU for {model_type}")
            device = torch.device("cpu")
            model = build_model(model_type, config).to(device)
        else:
            raise e

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

    val_metrics = evaluate(model, val_loader, device)

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

    avg_time, std_time = measure_inference_time(model, val_loader, device, n_batches=20)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("experiments/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("\nDone!")


if __name__ == "__main__":
    main()
