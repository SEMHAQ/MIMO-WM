"""
Round 14 experiments: 5 seeds, expanded dataset, MuJoCo MPC.
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
from src.models.mpc_controller import MPCController
from src.data.robot_dataset import create_dataloaders
from src.train.train import build_model, load_config, train_one_epoch, evaluate, measure_inference_time, count_parameters


def compute_r2_score(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def train_and_eval_model(model_type, config, device, seed):
    """Train and evaluate a model with given seed."""
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

    # Build model
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

    # Compute R² on full validation set
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

    # Multi-step prediction MSE (H=4, 8, 16)
    multi_step_mse = {}
    for H in [4, 8, 16]:
        mse_list = []
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                if states.shape[1] <= H:
                    continue
                future_actions = actions[:, -H:]
                init_states = states[:, :-H]
                init_actions = actions[:, :-H]
                target_traj = states[:, -H:]
                pred_traj = model.predict_trajectory(init_states, init_actions, future_actions)
                mse_val = nn.MSELoss()(pred_traj, target_traj).item()
                mse_list.append(mse_val)
        multi_step_mse[f"h{H}"] = np.mean(mse_list) if mse_list else float('inf')

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
        "multi_step_mse": multi_step_mse,
    }


def run_mpc_experiment(model_type, config, device, seed, n_episodes=5, n_steps=30):
    """Run MPC control experiment with a trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["train"]
    mpc_cfg = config["mpc"]

    # Train model first
    train_loader, val_loader = create_dataloaders(
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        num_workers=0,
        normalize=data_cfg["normalize"],
    )

    model = build_model(model_type, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    for epoch in range(train_cfg["epochs"]):
        train_one_epoch(model, train_loader, optimizer, config, device)
        scheduler.step()

    # MPC controller
    mpc = MPCController(
        world_model=model,
        horizon=mpc_cfg["horizon"],
        Q_weight=mpc_cfg["Q_weight"],
        R_weight=mpc_cfg["R_weight"],
        n_iterations=mpc_cfg["n_iterations"],
        lr=mpc_cfg["lr"],
        device=str(device),
    )

    # Run closed-loop control on validation episodes
    tracking_errors = []
    control_times = []

    val_episodes = list(val_loader.dataset)

    for ep_idx in range(min(n_episodes, len(val_episodes))):
        ep = val_episodes[ep_idx]
        states = ep["states"]
        actions = ep["actions"]

        if len(states) < data_cfg["seq_len"] + n_steps:
            continue

        # Use first seq_len states as history
        T = data_cfg["seq_len"]
        init_state = states[:T]
        init_action = actions[:T-1]

        # Target: state at T + n_steps
        target_idx = min(T + n_steps, len(states) - 1)
        target_state = states[target_idx]

        # Run MPC
        t0 = time.perf_counter()
        trajectory, executed_actions = mpc.closed_loop_control(
            init_state=init_state,
            init_actions=init_action,
            target_state=target_state,
            true_dynamics=None,  # Use model as dynamics
            n_steps=min(n_steps, len(states) - T),
        )
        t1 = time.perf_counter()

        # Compute tracking error
        steps_to_eval = min(len(trajectory), len(states) - T)
        for step in range(steps_to_eval):
            actual_idx = T + step
            if actual_idx < len(states):
                error = np.linalg.norm(trajectory[step] - states[actual_idx])
                tracking_errors.append(error)

        control_times.append((t1 - t0) / max(steps_to_eval, 1))

    return {
        "model": model_type,
        "seed": seed,
        "mean_tracking_error": float(np.mean(tracking_errors)) if tracking_errors else float('inf'),
        "std_tracking_error": float(np.std(tracking_errors)) if tracking_errors else 0.0,
        "mean_control_time_ms": float(np.mean(control_times) * 1000) if control_times else 0.0,
        "n_episodes_evaluated": min(n_episodes, len(val_episodes)),
    }


def run_sequence_length_experiment(model, val_loader, device, seq_lengths=[16, 32, 48, 64, 96, 128]):
    """Measure inference time and MSE at different sequence lengths."""
    results = {}
    model.eval()

    for seq_len in seq_lengths:
        times = []
        mses = []
        count = 0

        for batch in val_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            target = batch["target"].to(device)

            # Truncate or pad to desired seq_len
            if states.shape[1] > seq_len:
                states = states[:, -seq_len:]
                actions = actions[:, -(seq_len-1):]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(states, actions)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)
            mses.append(nn.MSELoss()(pred, target).item())
            count += 1
            if count >= 10:
                break

        results[seq_len] = {
            "infer_ms": float(np.mean(times)),
            "infer_std_ms": float(np.std(times)),
            "mse": float(np.mean(mses)),
        }

    return results


def run_batch_size_experiment(model, val_loader, device, batch_sizes=[1, 4, 16, 32, 64, 128]):
    """Measure inference time at different batch sizes."""
    results = {}
    model.eval()

    for bs in batch_sizes:
        times = []
        count = 0

        for batch in val_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)

            # Truncate batch
            if states.shape[0] > bs:
                states = states[:bs]
                actions = actions[:bs]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(states, actions)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)
            count += 1
            if count >= 10:
                break

        results[bs] = {
            "infer_ms": float(np.mean(times)),
            "infer_std_ms": float(np.std(times)),
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="all",
                       choices=["all", "main", "mpc", "seqlen", "batch"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("experiments/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1024]
    exp = args.experiment

    # ==================== Main Results (5 seeds) ====================
    if exp in ["all", "main"]:
        print("\n" + "="*60)
        print("  EXPERIMENT 1: Main Results (5 seeds)")
        print("="*60)

        all_seed_results = []
        for model_type in ["ssm", "lstm"]:
            for seed in seeds:
                print(f"\n  Training {model_type} with seed {seed}...")
                result = train_and_eval_model(model_type, config, device, seed)
                all_seed_results.append(result)
                print(f"    MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, R²={result['r2']:.4f}, "
                      f"Infer={result['infer_ms']:.1f}ms")

        with open(output_dir / "multi_seed_results_5seeds.json", "w") as f:
            json.dump(all_seed_results, f, indent=2)

        # Print summary
        print("\n  Summary (mean ± std over 5 seeds):")
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

    # ==================== MuJoCo MPC Experiment ====================
    if exp in ["all", "mpc"]:
        print("\n" + "="*60)
        print("  EXPERIMENT 2: MuJoCo MPC Control")
        print("="*60)

        mpc_results = []
        for model_type in ["ssm", "lstm"]:
            for seed in seeds[:3]:  # 3 seeds for MPC (expensive)
                print(f"\n  MPC with {model_type}, seed {seed}...")
                result = run_mpc_experiment(model_type, config, device, seed)
                mpc_results.append(result)
                print(f"    Tracking error: {result['mean_tracking_error']:.4f}±{result['std_tracking_error']:.4f}, "
                      f"Control time: {result['mean_control_time_ms']:.1f}ms")

        with open(output_dir / "mpc_results.json", "w") as f:
            json.dump(mpc_results, f, indent=2)

    # ==================== Sequence Length Experiment ====================
    if exp in ["all", "seqlen"]:
        print("\n" + "="*60)
        print("  EXPERIMENT 3: Sequence Length Sensitivity")
        print("="*60)

        # Train one model of each type
        seqlen_results = {}
        for model_type in ["ssm", "lstm"]:
            print(f"\n  Training {model_type} for sequence length experiment...")
            torch.manual_seed(42)
            np.random.seed(42)

            train_loader, val_loader = create_dataloaders(
                data_dir=config["data"]["data_dir"],
                seq_len=config["data"]["seq_len"],
                batch_size=config["data"]["batch_size"],
                num_workers=0,
                normalize=config["data"]["normalize"],
            )

            model = build_model(model_type, config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["train"]["lr"],
                weight_decay=config["train"]["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["train"]["epochs"]
            )

            for epoch in range(config["train"]["epochs"]):
                train_one_epoch(model, train_loader, optimizer, config, device)
                scheduler.step()

            seq_results = run_sequence_length_experiment(model, val_loader, device)
            seqlen_results[model_type] = seq_results
            print(f"    {model_type} sequence length results: {seq_results}")

        with open(output_dir / "seqlen_results.json", "w") as f:
            json.dump(seqlen_results, f, indent=2)

    # ==================== Batch Size Experiment ====================
    if exp in ["all", "batch"]:
        print("\n" + "="*60)
        print("  EXPERIMENT 4: Batch Size Inference")
        print("="*60)

        batch_results = {}
        for model_type in ["ssm", "lstm"]:
            print(f"\n  Training {model_type} for batch size experiment...")
            torch.manual_seed(42)
            np.random.seed(42)

            train_loader, val_loader = create_dataloaders(
                data_dir=config["data"]["data_dir"],
                seq_len=config["data"]["seq_len"],
                batch_size=config["data"]["batch_size"],
                num_workers=0,
                normalize=config["data"]["normalize"],
            )

            model = build_model(model_type, config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["train"]["lr"],
                weight_decay=config["train"]["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["train"]["epochs"]
            )

            for epoch in range(config["train"]["epochs"]):
                train_one_epoch(model, train_loader, optimizer, config, device)
                scheduler.step()

            bs_results = run_batch_size_experiment(model, val_loader, device)
            batch_results[model_type] = bs_results
            print(f"    {model_type} batch size results: {bs_results}")

        with open(output_dir / "batch_results.json", "w") as f:
            json.dump(batch_results, f, indent=2)

    print("\n" + "="*60)
    print("  All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
