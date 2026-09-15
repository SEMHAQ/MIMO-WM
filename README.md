# MIMO-WM: A Lightweight State Space World Model for Humanoid Robots

**MIMO-WM: 面向人形机器人的轻量级状态空间世界模型**

[![Paper](https://img.shields.io/badge/Paper-CTA%202026-blue)](https://github.com/SEMHAQ/MIMO-WM)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

MIMO-WM is a lightweight world model based on multi-input multi-output state space model (MIMO-SSM) architecture for humanoid robot state prediction. Each input dimension maintains an independent state space through parallel scanning, with a sigmoid gating mechanism that dynamically adjusts information flow. The model reaches the best prediction accuracy among lightweight models while requiring only 0.138M parameters.

## Key Results

### State Prediction Performance (T=32, 5 seeds, mean ± std)

MSE in ×10⁻². All models share the hidden size $D{=}96$, $L{=}2$ for a like-for-like comparison.

**Humanoid**

| Model | MSE (×10⁻²) | R² | Params (M) |
|-------|-------------|-----|------------|
| LSTM-WM | 39.93±0.36 | 0.501 | 0.227 |
| GRU-WM | 36.60±0.30 | 0.542 | 0.190 |
| Transformer-WM | 28.11±0.72 | 0.648 | 0.302 |
| Mamba-WM | 20.18±0.24 | 0.748 | 0.224 |
| TCN-WM | 20.68±0.32 | 0.741 | 0.189 |
| **MIMO-WM** | **19.87±0.23** | **0.751** | **0.138** |
| MIMO-WM (openGate init) | 19.55±0.22 | 0.755 | 0.138 |
| MIMO-WM (w/o gating) | 20.55±0.05 | 0.743 | 0.101 |
| S4D-WM | 31.18±0.38 | 0.610 | 0.101 |
| LRU-WM | 20.42±0.11 | 0.745 | 0.101 |
| Performer-WM | 20.99±0.18 | 0.737 | 0.162 |
| Transformer-WM (regular scale) | 25.13±0.54 | 0.686 | 1.509 |

**HumanoidStandup**

| Model | MSE (×10⁻²) | R² | Params (M) |
|-------|-------------|-----|------------|
| **MIMO-WM** | **53.10±0.07** | **0.444** | **0.138** |
| MIMO-WM (openGate init) | 52.71±0.08 | 0.449 | 0.138 |
| MIMO-WM (w/o gating) | 50.61±0.22 | 0.470 | 0.101 |
| S4D-WM | 51.90±0.17 | 0.457 | 0.101 |
| LRU-WM | 50.35±0.03 | 0.473 | 0.101 |
| Performer-WM | 53.95±0.42 | 0.436 | 0.162 |
| Transformer-WM (regular scale) | 54.53±0.28 | 0.429 | 1.509 |

"Regular scale" Transformer uses $D{=}192$, 6 heads, $L{=}3$, FFN$=4D$ (1.509M parameters), included to
test whether the ~0.3M baselines were under-parameterised. Per-seed values and the evaluation scripts
are in [`revision_experiments/`](revision_experiments/).

### Highlights

- **Best accuracy among lightweight models**: MSE 19.87×10⁻² on Humanoid at only 0.138M parameters,
  ahead of Mamba-WM (20.18) and TCN-WM (20.68).
- **A ~11× larger Transformer does not close the gap**: regular-scale Transformer (1.509M) reaches
  25.13×10⁻² on Humanoid, versus 19.87 for MIMO-WM.
- **Gating pays off where the dynamics are coupled**: removing it costs 3.3% on Humanoid, but on
  HumanoidStandup it is not necessary — the benefit depends on the task dynamics.
- **Deployable**: pure real-valued recurrence, ONNX-exportable, verified equal to the training
  convolution path to ~2.4×10⁻⁷; 2.69 ms per 8-step window on an ARM board (ONNXRuntime, CPU),
  with a 9.6 MB resident-memory increase and a 0.59 MB model.

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

Experiments use MuJoCo medium datasets for Humanoid (348-dim state, 17-dim action) and HumanoidStandup (348-dim state, 17-dim action), collected via Gymnasium.

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

revision_experiments/
  README.md             # Supplementary material: file ↔ table/section map, repro commands
  results/              # Raw outputs (JSON) and exported ONNX model
  scripts/              # Evaluation scripts generating the results above
```

## Supplementary Material

Results, evaluation scripts and the exported ONNX model supporting the revision of the manuscript
are under [`revision_experiments/`](revision_experiments/). See
[`revision_experiments/README.md`](revision_experiments/README.md) for the mapping from each result
file to the corresponding table/section of the manuscript, along with reproduction commands.

## Citation

```bibtex
@article{mimo-wm2026,
  title={MIMO-WM: A Lightweight State Space World Model for Humanoid Robots},
  author={Zhou, Xin-min and Yu, Huan-jie and Zhang, Hui-hui and Wang, Wei and Chen, Lu},
  journal={Control Theory \& Applications},
  year={2026}
}
```

## License

MIT License
