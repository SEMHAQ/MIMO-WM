#!/usr/bin/env python3
"""Generate sequence length sensitivity figure with D4RL data."""
import matplotlib.pyplot as plt
import numpy as np
import json

# Load results
with open('experiments/seqlen_results_final.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors
colors = {
    'humanoid': '#2E86AB',
    'ant': '#A23B72'
}

# Plot MSE
ax1 = axes[0]
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    mse_values = [r['mse'] for r in data]
    ax1.plot(T_values, mse_values, 'o-', color=colors[dataset], 
             label=f'{dataset.capitalize()} ({348 if dataset == "humanoid" else 105}D)',
             linewidth=2, markersize=8)

ax1.set_xlabel('Sequence Length (T)', fontsize=12)
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_title('Prediction MSE vs Sequence Length', fontsize=14)
ax1.legend(fontsize=11)
ax1.set_xscale('log', base=2)
ax1.set_xticks([16, 32, 64, 128, 256])
ax1.set_xticklabels(['16', '32', '64', '128', '256'])
ax1.grid(True, alpha=0.3)

# Plot R²
ax2 = axes[1]
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    r2_values = [r['r2'] for r in data]
    ax2.plot(T_values, r2_values, 'o-', color=colors[dataset],
             label=f'{dataset.capitalize()} ({348 if dataset == "humanoid" else 105}D)',
             linewidth=2, markersize=8)

ax2.set_xlabel('Sequence Length (T)', fontsize=12)
ax2.set_ylabel('R²', fontsize=12)
ax2.set_title('R² Score vs Sequence Length', fontsize=14)
ax2.legend(fontsize=11)
ax2.set_xscale('log', base=2)
ax2.set_xticks([16, 32, 64, 128, 256])
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/seqlen_sensitivity.png', dpi=300, bbox_inches='tight')
print("Generated seqlen_sensitivity.pdf")

# Also generate inference time figure
fig2, ax3 = plt.subplots(figsize=(8, 5))

for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    infer_values = [r['infer_ms'] for r in data]
    ax3.plot(T_values, infer_values, 'o-', color=colors[dataset],
             label=f'{dataset.capitalize()} ({348 if dataset == "humanoid" else 105}D)',
             linewidth=2, markersize=8)

ax3.set_xlabel('Sequence Length (T)', fontsize=12)
ax3.set_ylabel('Inference Time (ms)', fontsize=12)
ax3.set_title('Inference Time vs Sequence Length', fontsize=14)
ax3.legend(fontsize=11)
ax3.set_xscale('log', base=2)
ax3.set_xticks([16, 32, 64, 128, 256])
ax3.set_xticklabels(['16', '32', '64', '128', '256'])
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10ms threshold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/inference_vs_seqlen.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/inference_vs_seqlen.png', dpi=300, bbox_inches='tight')
print("Generated inference_vs_seqlen.pdf")
