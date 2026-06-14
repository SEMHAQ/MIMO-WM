#!/usr/bin/env python3
"""Generate main comparison figure with D4RL results."""
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['LSTM-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM']
humanoid_mse = [0.367, 0.278, 0.259, 0.245]
ant_mse = [0.800, 0.718, 0.746, 0.728]
humanoid_params = [0.64, 0.15, 0.66, 0.23]
ant_params = [0.57, 0.12, 0.59, 0.16]

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# Plot 1: MSE comparison
ax1 = axes[0]
x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, humanoid_mse, width, label='Humanoid (348D)', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, ant_mse, width, label='Ant (105D)', color='#A23B72', alpha=0.8)

ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_title('Prediction MSE on D4RL Datasets', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.legend(fontsize=11)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Plot 2: Parameter efficiency
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, humanoid_params, width, label='Humanoid (348D)', color='#2E86AB', alpha=0.8)
bars4 = ax2.bar(x + width/2, ant_params, width, label='Ant (105D)', color='#A23B72', alpha=0.8)

ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Parameters (M)', fontsize=12)
ax2.set_title('Parameter Count Comparison', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.legend(fontsize=11)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}M',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}M',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Highlight S4D-WM
for ax in axes:
    ax.patches[-1].set_edgecolor('black')
    ax.patches[-1].set_linewidth(2)
    ax.patches[-2].set_edgecolor('black')
    ax.patches[-2].set_linewidth(2)

plt.tight_layout()
plt.savefig('paper/figures/main_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/main_comparison.png', dpi=300, bbox_inches='tight')
print("Generated main_comparison.pdf")
