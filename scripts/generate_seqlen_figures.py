#!/usr/bin/env python3
"""Generate sequence length sensitivity figure - top-bottom layout, Chinese labels."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
import numpy as np
import json

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

with open('experiments/seqlen_results_final.json', 'r') as f:
    results = json.load(f)

colors = {'humanoid': '#2E86AB', 'ant': '#A23B72'}
labels = {'humanoid': 'Humanoid (348D)', 'ant': 'Ant (105D)'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.0))

# Top: MSE
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    mse_values = [r['mse'] for r in data]
    ax1.plot(T_values, mse_values, 'o-', color=colors[dataset],
             label=labels[dataset], linewidth=1.5, markersize=5)

# Recommended zones
ax1.axvspan(16, 32, alpha=0.12, color='#2E86AB')
ax1.annotate('推荐区域', xy=(18, ax1.get_ylim()[1]*0.9), fontproperties=zhfont_s,
             fontsize=8, color='#2E86AB', fontweight='bold')

ax1.set_xlabel('序列长度 T', fontproperties=zhfont)
ax1.set_ylabel('MSE', fontsize=10)
ax1.legend(fontsize=8.5, prop=zhfont_s)
ax1.set_xscale('log', base=2)
ax1.set_xticks([16, 32, 64, 128, 256])
ax1.set_xticklabels(['16', '32', '64', '128', '256'])
ax1.grid(True, alpha=0.3)
ax1.set_title('(a) 预测MSE随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

# Bottom: R2
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    r2_values = [r['r2'] for r in data]
    ax2.plot(T_values, r2_values, 'o-', color=colors[dataset],
             label=labels[dataset], linewidth=1.5, markersize=5)

ax2.axvspan(128, 256, alpha=0.12, color='#A23B72')
ax2.annotate('推荐区域', xy=(130, ax2.get_ylim()[1]*0.9), fontproperties=zhfont_s,
             fontsize=8, color='#A23B72', fontweight='bold')

ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
ax2.set_ylabel('R2', fontsize=10)
ax2.legend(fontsize=8.5, prop=zhfont_s)
ax2.set_xscale('log', base=2)
ax2.set_xticks([16, 32, 64, 128, 256])
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b) R2随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

plt.tight_layout()
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
