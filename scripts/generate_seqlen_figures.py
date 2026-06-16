#!/usr/bin/env python3
"""SeqLen sensitivity: top=MSE, bottom=R², both datasets per subplot."""
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.2))

# ---- Top: MSE ----
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    mse_values = [r['mse'] for r in data]
    ax1.plot(T_values, mse_values, 'o-', color=colors[dataset],
             label=labels[dataset], linewidth=1.5, markersize=5, zorder=3)

# Recommendation zones
ax1.axvspan(14, 36, alpha=0.10, color=colors['humanoid'], zorder=1)
ax1.text(15, ax1.get_ylim()[1] * 0.92, 'Humanoid推荐',
         fontproperties=zhfont_s, fontsize=7.5, color=colors['humanoid'],
         fontweight='bold', va='top')
ax1.axvspan(100, 300, alpha=0.10, color=colors['ant'], zorder=1)
ax1.text(105, ax1.get_ylim()[1] * 0.92, 'Ant推荐',
         fontproperties=zhfont_s, fontsize=7.5, color=colors['ant'],
         fontweight='bold', va='top')

ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xscale('log', base=2)
ax1.set_xticks([16, 32, 64, 128, 256])
ax1.set_xticklabels([])
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) 预测MSE随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

# ---- Bottom: R² ----
for dataset, data in results.items():
    T_values = [r['T'] for r in data]
    r2_values = [r['r2'] for r in data]
    ax2.plot(T_values, r2_values, 'o-', color=colors[dataset],
             label=labels[dataset], linewidth=1.5, markersize=5, zorder=3)

# Recommendation zones
ax2.axvspan(14, 36, alpha=0.10, color=colors['humanoid'], zorder=1)
ax2.text(15, ax2.get_ylim()[1] * 0.92, 'Humanoid推荐',
         fontproperties=zhfont_s, fontsize=7.5, color=colors['humanoid'],
         fontweight='bold', va='top')
ax2.axvspan(100, 300, alpha=0.10, color=colors['ant'], zorder=1)
ax2.text(105, ax2.get_ylim()[1] * 0.92, 'Ant推荐',
         fontproperties=zhfont_s, fontsize=7.5, color=colors['ant'],
         fontweight='bold', va='top')

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.6)
ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
ax2.set_ylabel('R$^2$', fontsize=10)
ax2.set_xscale('log', base=2)
ax2.set_xticks([16, 32, 64, 128, 256])
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) R²随序列长度的变化', fontproperties=zhfont, fontsize=10, pad=8)

# Shared legend below bottom x-axis
handles, lbls = ax1.get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=2, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True,
           framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
