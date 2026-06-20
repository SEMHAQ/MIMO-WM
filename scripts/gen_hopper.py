"""Hopper (11D) performance comparison bar chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

models = ['LSTM', 'Trans.', 'GRU', 'Mamba', 'S4D-WM']
mse = [1.39, 1.30, 1.39, 1.47, 1.33]
errs = [0.01, 0.02, 0.04, 0.01, 0.03]
colors = ['#B0B0B0', '#B0B0B0', '#B0B0B0', '#B0B0B0', '#1f77b4']

fig, ax = plt.subplots(figsize=(4.5, 3.2))

bars = ax.bar(range(len(models)), mse, yerr=errs, color=colors,
              edgecolor='none', capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('MSE (×10⁻²)', fontsize=10)
ax.set_title('D4RL Hopper (11维) 性能对比', fontproperties=zhfont, fontsize=10, pad=8)
ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Highlight S4D-WM
bars[4].set_edgecolor('#1f77b4')
bars[4].set_linewidth(1.5)

for bar, val in zip(bars, mse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/hopper_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: hopper_comparison.pdf")
