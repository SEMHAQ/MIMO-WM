# MS-WM: Multi-Scale Dynamics World Model for High-Dimensional Robot State Prediction

**面向高维机器人状态预测的多尺度动力学世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/SSM-World-Model)
[![Code](https://img.shields.io/badge/Code-Python-green)](https://github.com/SEMHAQ/SSM-World-Model)

## Overview

MS-WM is a multi-scale dynamics world model that separately handles slow dynamics (position), fast dynamics (velocity), and instantaneous dynamics (forces) through three specialized branches: slow SSM, fast SSM, and local attention. This physics-inspired design achieves state-of-the-art accuracy on high-dimensional robot state prediction while maintaining lightweight parameters.

## Key Results

### Performance Comparison (T=32, 5 seeds)

| Model | Humanoid (348D) | Ant (105D) |
|-------|-----------------|------------|
| LSTM-WM | 40.87±0.63 | 85.51±1.36 |
| GRU-WM | 35.60±0.65 | 81.91±3.65 |
| Transformer-WM | 25.85±0.35 | 73.78±1.59 |
| Mamba-WM | 27.34±0.44 | 78.52±1.19 |
| S4D-WM | 27.13±0.35 | 77.02±0.43 |
| **MS-WM** | **20.74±0.28** | **72.59±0.71** |

**Key Findings**:
- MS-WM achieves **20% lower MSE** than Transformer on Humanoid (348D)
- MS-WM achieves **1.6% lower MSE** than Transformer on Ant (105D)
- MS-WM has **0.019M parameters** (lightweight)
- Multi-scale modeling is more effective for high-dimensional complex dynamics

## Dataset

We use [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/) medium-v0 datasets:

```bash
# Download and prepare datasets
python scripts/download_all_data.py
```

This will download:
- `humanoid/` — Gymnasium Humanoid-medium (348D state, 17D action)
- `ant/` — Gymnasium Ant-medium (105D state, 8D action)

## Project Structure

```
SSM-World-Model/
├── src/
│   ├── models/
│   │   ├── ssm_world_model.py     # S4D baseline
│   │   ├── baselines.py           # LSTM, GRU, Transformer baselines
│   │   ├── mamba_world_model.py   # Mamba baseline
│   │   └── mpc_controller.py      # Model Predictive Control
│   └── train/
│       └── train.py               # Training loop
├── scripts/
│   ├── download_all_data.py       # Download Gymnasium MuJoCo datasets
│   ├── run_all_experiments.py     # Full experiment pipeline (all models, all datasets, 5 seeds)
│   └── update_paper_tables.py     # Generate LaTeX tables from results
├── experiments/
│   └── all_results.json           # All experiment results
├── data/
│   ├── humanoid/                  # Gymnasium Humanoid-medium (348D)
│   └── ant/                       # Gymnasium Ant-medium (105D)
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
@article{zhou2026mswm,
  title   = {面向高维机器人状态预测的多尺度动力学世界模型},
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
