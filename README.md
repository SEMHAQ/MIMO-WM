# SSM-World-Model

**A Lightweight State Space World Model for Humanoid Robot State Prediction and MPC Control**

This repository contains the source code for the paper:

> **面向人形机器人状态预测的轻量级状态空间世界模型**
>
> 周新民, 余焕杰 (湖南工商大学 / 湘江实验室)
>
> *控制理论与应用 (Control Theory & Applications)*, 2026

## Highlights

- **Lightweight SSM-based world model** (SSM-WM) using diagonal state space parameterization (S4D-style) with Mamba-style gating
- **O(T log T) training complexity**, O(1) single-step inference latency
- **~7× speedup** over LSTM world models in batch inference scenarios
- On MuJoCo Humanoid dataset: **6% better MSE than LSTM-WM**, **13% better than Transformer-WM**, comparable to Mamba-WM (<2% gap)
- **MPC integration**: 5.1Hz control frequency (synthetic), 2.1Hz (MuJoCo Humanoid)

## Project Structure

```
SSM-World-Model/
├── src/
│   ├── models/
│   │   ├── ssm_world_model.py    # Core SSM-WM architecture
│   │   ├── baselines.py          # LSTM-WM, Transformer-WM, Mamba-WM
│   │   └── mpc_controller.py     # Model Predictive Control integration
│   ├── data/
│   │   └── robot_dataset.py      # Dataset loading & preprocessing
│   ├── train/
│   │   └── train.py              # Training loop
│   └── utils/
│       └── helpers.py            # Utility functions
├── scripts/
│   ├── generate_figures_cn.py    # Reproduce paper figures (Chinese labels)
│   ├── generate_mujoco_data.py   # Generate MuJoCo Humanoid dataset
│   ├── generate_expanded_dataset.py
│   ├── run_full_experiments.py   # Full experiment pipeline
│   └── quick_test.py             # Quick sanity check
├── configs/
│   ├── default.yaml              # Default config (synthetic dataset)
│   └── mujoco.yaml               # MuJoCo Humanoid config
├── experiments/
│   └── paper_results/            # Final experiment results (JSON)
└── paper/
    ├── main.tex                  # Paper source (LaTeX)
    ├── main.pdf                  # Compiled PDF
    └── figures/                  # Paper figures (PDF, EPS, PNG)
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch 2.0+
- NumPy, einops, matplotlib, tqdm, tensorboard

### Generate MuJoCo Dataset

```bash
python scripts/generate_mujoco_data.py
```

### Train SSM-WM

```bash
# Synthetic dataset
python src/train/train.py --config configs/default.yaml

# MuJoCo Humanoid
python src/train/train.py --config configs/mujoco.yaml
```

### Run Full Experiments

```bash
python scripts/run_full_experiments.py
```

### Reproduce Paper Figures

```bash
python scripts/generate_figures_cn.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026ssmwm,
  title   = {面向人形机器人状态预测的轻量级状态空间世界模型},
  author  = {周新民 and 余焕杰},
  journal = {控制理论与应用},
  year    = {2026}
}
```

## Acknowledgments

This work was supported by the National Social Science Fund of China (Grant No. 21BGL231) and the Xiangjiang Laboratory (Grant No. 23XJ01001).

## License

This project is for academic research purposes.
