# MIMO-WM: A Lightweight World Model Based on Multi-Input Multi-Output State Space Model for Humanoid Robots

**基于多输入多输出状态空间模型的人形机器人轻量级世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/MIMO-WM)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

MIMO-WM is a lightweight world model based on multi-input multi-output state space model (MIMO-SSM) architecture for humanoid robot state prediction. Each input dimension maintains an independent state space through parallel scanning, with a sigmoid gating mechanism that dynamically adjusts information flow. The model achieves state-of-the-art prediction accuracy while requiring only 0.138M parameters.

## Key Results

### State Prediction Performance (T=32, 5 seeds)

| Model | Humanoid MSE (×10⁻²) | Humanoid R² | Params (M) |
|-------|----------------------|-------------|------------|
| LSTM-WM | 39.93±0.36 | 0.501 | 0.227 |
| GRU-WM | 36.60±0.30 | 0.542 | 0.190 |
| Transformer-WM | 28.11±0.72 | 0.648 | 0.302 |
| Mamba-WM | 20.18±0.24 | 0.748 | 0.224 |
| TCN-WM | 20.68±0.32 | 0.741 | 0.189 |
| **MIMO-WM** | **19.87±0.23** | **0.751** | **0.138** |

### Highlights

- **Best accuracy**: MSE 19.87×10⁻² on Humanoid, outperforming Mamba-WM by 1.5%
- **Most lightweight**: Only 0.138M parameters, 38% fewer than Mamba-WM
- **Theoretical guarantees**: Proven dual-mode equivalence, complexity advantage, and CEM-MPC convergence

## Architecture

```
Input [s; a] → Encoder → [MIMO Block × L] → Decoder → ŝ
                              ↑
                    LayerNorm → DiagSSM → Gate(σ) → Residual
```

- **MIMO-SSM**: D parallel diagonal SSMs, one per input dimension
- **Gating**: Sigmoid mechanism for adaptive information control
- **Dual-mode**: Convolution (O(T log T)) for training, recurrent (O(1)) for deployment

## Dataset

Experiments use MuJoCo medium datasets for Humanoid (348-dim state, 17-dim action) and HumanoidStandup (376-dim state, 17-dim action), collected via Gymnasium.

[📥 Download Dataset (Google Drive)](https://drive.google.com/drive/folders/13k6u48Iu3vNW0nebvZ4RgT6M6nhoUorX?usp=drive_link)

Place the downloaded `data/` folder under the project root.

## Quick Start

### Installation

```bash
git clone https://github.com/SEMHAQ/MIMO-WM.git
cd MIMO-WM
pip install torch numpy matplotlib
```

### Train & Evaluate

```bash
# State prediction (Humanoid + HumanoidStandup)
python3 scripts/run_exp1_state_prediction.py

# Ablation study
python3 scripts/run_ablation_mimo.py

# Sequence length sensitivity
python3 scripts/run_seqlen_sensitivity.py
python3 scripts/run_seqlen_standup.py

# MPC planning
python3 scripts/run_exp4_mpc.py
```

### Generate Figures

```bash
python3 scripts/gen_figures.py   # Ablation + sequence length
python3 scripts/gen_radar.py     # Radar comparison
```

## Project Structure

```
src/models/
  mimo_world_model.py    # MIMO-WM model (MIMOLayer + MIMOWorldModel)
  ssm_world_model.py     # DiagSSM core (diagonal SSM with conv/recurrent modes)
  baselines.py           # LSTM, GRU, Transformer, TCN baselines
  mamba_world_model.py   # Mamba baseline

scripts/
  run_exp1_state_prediction.py   # Experiment 1: state prediction
  run_ablation_mimo.py           # Ablation study
  run_seqlen_sensitivity.py      # Humanoid sequence length analysis
  run_seqlen_standup.py          # HumanoidStandup sequence length analysis
  run_exp4_mpc.py                # MPC planning experiment
  gen_figures.py                 # Figure generation
  gen_radar.py                   # Radar chart generation

paper/
  main.tex              # Main paper (CTA format)
  kzllyyhead.tex        # CTA template header
```

## Documentation

Full documentation is available via MkDocs.

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Citation

```bibtex
@article{mimo-wm2026,
  title={A Lightweight World Model Based on Multi-Input Multi-Output State Space Model for Humanoid Robots},
  author={Zhou, Xin-min and Yu, Huan-jie and Zhang, Hui-hui and Wang, Wei and Chen, Lu},
  journal={Control Theory \& Applications},
  year={2026}
}
```

## License

MIT License
