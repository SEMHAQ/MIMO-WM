# FSM-WM: Fusion State Space World Model for High-Dimensional Robot State Prediction

**面向高维机器人状态预测的融合状态空间世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/FSM-WM)
[![Code](https://img.shields.io/badge/Code-Python-green)](https://github.com/SEMHAQ/FSM-WM)

## Overview

FSM-WM is a lightweight world model based on Fusion State-space Model (FSM), a novel SSM-Attention hybrid architecture that adaptively fuses SSM's long-range modeling with local attention's fine-grained feature extraction through a learned gating mechanism. It achieves **O(T log T) training complexity** and **O(1) single-step inference latency**.

## Key Results

### Performance Comparison (T=32, 5 seeds)

| Dataset | Dimension | FSM-WM | Best Baseline | Improvement |
|---------|-----------|--------|---------------|-------------|
| Humanoid | 376D | **24.85±0.44** | 25.95 (Trans.) | -4.2% |
| Ant | 105D | 76.65±0.39 | **72.97** (Trans.) | +5.0% |
| Walker2d | 17D | **2.89±0.03** | 2.80 (Trans.) | +3.2% |

**Key Findings**:
- FSM-WM achieves best accuracy on high-dimensional Humanoid and low-dimensional Walker2d
- On medium-dimensional Ant, Transformer-WM is slightly better
- FSM's adaptive gating mechanism effectively combines SSM and attention strengths

## Dataset

We use [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/) expert-v0 datasets:

```bash
# Download and prepare datasets
python scripts/download_all_data.py
```

This will download:
- `humanoid/` — Gymnasium Humanoid-expert (376D state, 17D action)
- `ant/` — Gymnasium Ant-expert (105D state, 8D action)
- `walker2d/` — Gymnasium Walker2d-expert (17D state, 6D action)

## Project Structure

```
FSM-WM/
├── src/
│   ├── models/
│   │   ├── fusion_ssm.py          # FSM-WM architecture (SSM + Attention hybrid)
│   │   ├── ssm_world_model.py     # S4D-WM baseline
│   │   ├── baselines.py           # LSTM-WM, Trans.-WM, GRU-WM
│   │   ├── mamba_world_model.py   # Mamba-WM baseline
│   │   └── mpc_controller.py      # Model Predictive Control integration
│   └── train/
│       └── train.py               # Training loop
├── scripts/
│   ├── download_all_data.py       # Download Gymnasium MuJoCo datasets
│   ├── run_all_experiments.py     # Full experiment pipeline (all models, all datasets, 5 seeds)
│   └── update_paper_tables.py     # Generate LaTeX tables from results
├── experiments/
│   └── all_results.json           # All experiment results
├── data/
│   ├── humanoid/                  # Gymnasium Humanoid-expert (376D)
│   ├── ant/                       # Gymnasium Ant-expert (105D)
│   └── walker2d/                  # Gymnasium Walker2d-expert (17D)
└── paper/
    ├── main.tex                   # Paper source (LaTeX, CTA template)
    └── figures/                   # Paper figures
```

## Quick Start

### Requirements

```bash
pip install torch numpy minari gymnasium matplotlib tqdm
```

### Download Dataset

```bash
python scripts/download_all_data.py
```

### Run Experiments

```bash
# Full experiment pipeline (all models, all datasets, 5 seeds)
python scripts/run_all_experiments.py

# Update paper tables with results
python scripts/update_paper_tables.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026fsmwm,
  title   = {面向高维机器人状态预测的融合状态空间世界模型},
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
