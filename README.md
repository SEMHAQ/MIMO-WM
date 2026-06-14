# S4D-WM: Lightweight State Space World Model for Joint State Prediction in Embodied Intelligence

**面向具身智能关节状态预测的轻量级状态空间世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/S4D-WM)
[![Code](https://img.shields.io/badge/Code-Python-green)](https://github.com/SEMHAQ/S4D-WM)
[![Dataset](https://img.shields.io/badge/Dataset-D4RL-orange)](https://drive.google.com/drive/folders/16TBWD3CeWEmL3Al1M9goMzYN8TDJgKyi?usp=drive_link)

## Overview

S4D-WM is a lightweight world model based on diagonal state space models (S4D-style) with Mamba-style gating blocks for joint state prediction in embodied intelligence. It achieves **O(T log T) training complexity** and **O(1) single-step inference latency**.

## Key Results

### D4RL Humanoid Dataset (348 dimensions)

| Model | MSE | R² | Inference (ms) | Params (M) |
|-------|-----|-----|----------------|------------|
| LSTM-WM | 0.367 | 0.541 | 2.5 | 0.64 |
| Transformer-WM | 0.278 | 0.653 | 1.6 | 0.15 |
| Mamba-WM | 0.259 | 0.676 | 3.5 | 0.66 |
| **S4D-WM** | **0.245** | **0.694** | **3.4** | **0.23** |

- **33% better MSE than LSTM-WM**
- **12% better MSE than Transformer-WM**
- **6% better MSE than Mamba-WM**
- **Only 0.23M parameters** (36% of LSTM-WM)

### D4RL Ant Dataset (105 dimensions)

| Model | MSE | R² | Inference (ms) | Params (M) |
|-------|-----|-----|----------------|------------|
| LSTM-WM | 0.800 | 0.066 | 0.8 | 0.57 |
| Transformer-WM | 0.718 | 0.162 | 1.6 | 0.12 |
| Mamba-WM | 0.746 | 0.129 | 3.4 | 0.59 |
| **S4D-WM** | **0.728** | **0.150** | **3.6** | **0.16** |

- **Comparable to Transformer-WM** (only 1.4% gap)
- **Best parameter efficiency** (0.16M, 28% of LSTM-WM)

### Sequence Length Sensitivity

| T | Humanoid MSE | Humanoid R² | Ant MSE | Ant R² |
|---|--------------|-------------|---------|--------|
| 16 | **0.291** | **0.656** | 0.542 | 0.302 |
| 32 | 0.442 | 0.479 | 0.728 | 0.150 |
| 64 | 0.612 | 0.153 | 0.942 | -0.019 |
| 128 | 1.213 | -0.623 | 0.934 | 0.139 |
| 256 | 2.146 | -1.694 | **0.480** | **0.131** |

**Key Finding**: Optimal sequence length depends on dataset dimensionality:
- **High-dimensional (348D)**: Shorter sequences (T=16) work best
- **Low-dimensional (105D)**: Longer sequences (T=256) work best

## Dataset

The D4RL datasets used in this paper are available at:

**[Google Drive: D4RL Datasets](https://drive.google.com/drive/folders/16TBWD3CeWEmL3Al1M9goMzYN8TDJgKyi?usp=drive_link)**

Contents:
- `humanoid/` - D4RL Humanoid-medium (348 dimensions, 1163 episodes)
- `ant/` - D4RL Ant-medium (105 dimensions, 1047 episodes)

## Project Structure

```
S4D-WM/
├── src/
│   ├── models/
│   │   ├── ssm_world_model.py    # S4D-WM architecture
│   │   ├── baselines.py          # LSTM-WM, Transformer-WM, Mamba-WM
│   │   └── mpc_controller.py     # Model Predictive Control integration
│   └── train/
│       └── train.py              # Training loop
├── scripts/
│   ├── train_all_v2.py           # Train all models on D4RL datasets
│   ├── train_seqlen_final.py     # Sequence length sensitivity analysis
│   └── generate_figures_nature.py # Generate paper figures
├── experiments/
│   ├── final_results.json        # Final experiment results
│   └── seqlen_results_final.json # Sequence length results
├── data/
│   ├── humanoid/                 # D4RL Humanoid-medium (348D)
│   └── ant/                      # D4RL Ant-medium (105D)
└── paper/
    ├── main.tex                  # Paper source (LaTeX)
    ├── main.pdf                  # Compiled PDF
    └── figures/                  # Paper figures
```

## Quick Start

### Requirements

```bash
pip install torch numpy matplotlib tqdm
```

### Download Dataset

```bash
# Download from Google Drive link above
# Place in data/ directory
```

### Train Models

```bash
# Train all models on D4RL datasets
python scripts/train_all_v2.py

# Sequence length sensitivity analysis
python scripts/train_seqlen_final.py
```

### Generate Figures

```bash
python scripts/generate_figures_nature.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026s4dwm,
  title   = {面向具身智能关节状态预测的轻量级状态空间世界模型},
  author  = {周新民 and 余焕杰 and 张慧慧 and 王伟 and 陈露},
  journal = {控制理论与应用},
  year    = {2026}
}
```

## Acknowledgments

This work was supported by:
- National Social Science Fund of China (Grant No. 21BGL231)
- Major Program of Xiangjiang Laboratory (Grant No. 24XJ01001; 25XJ01001)

## License

This project is for academic research purposes.

## Contact

- 周新民: zhouxinmin2699@163.com
- 余焕杰: semhaqx@gmail.com
- 张慧慧: huihuiz054@gmail.com
