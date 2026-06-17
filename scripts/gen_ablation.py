"""Regenerate ablation figure with D4RL Humanoid data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# D4RL Humanoid ablation data (Table 5) - multi-seed results
configs = ['默认', '无门控', '无残差', 'L=2', 'L=6', 'N=32', 'D=64', 'D=256']
mse_vals = [0.248, 0.263, 0.262, 0.247, 0.246, 0.244, 0.301, 0.234]
mse_errs = [0.003, 0.004, 0.002, 0.003, 0.005, 0.002, 0.006, 0.003]
r2_vals = [0.690, 0.671, 0.672, 0.691, 0.693, 0.695, 0.623, 0.707]
params = [0.23, 0.16, 0.23, 0.12, 0.28, 0.26, 0.09, 0.65]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (a) MSE with error bars
colors_mse = ['#1f77b4' if i == 0 else '#aec7e8' for i in range(len(configs))]
bars = ax1.bar(range(len(configs)), mse_vals, yerr=mse_errs, color=colors_mse, 
               alpha=0.85, edgecolor='none', capsize=3, error_kw={'linewidth': 0.8})
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(configs, fontproperties=zhfont, fontsize=8, rotation=15)
ax1.set_ylabel('MSE', fontsize=10)
ax1.set_title('(a) 预测MSE', fontproperties=zhfont, fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
# Highlight default
bars[0].set_edgecolor('#1f77b4')
bars[0].set_linewidth(1.5)

# (b) Params
colors_p = ['#1f77b4' if i == 0 else '#aec7e8' for i in range(len(configs))]
bars2 = ax2.bar(range(len(configs)), params, color=colors_p, alpha=0.85, edgecolor='none')
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, fontproperties=zhfont, fontsize=8, rotation=15)
ax2.set_ylabel('参数量 (M)', fontsize=10)
ax2.set_title('(b) 参数量', fontproperties=zhfont, fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
bars2[0].set_edgecolor('#1f77b4')
bars2[0].set_linewidth(1.5)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/ablation_results.pdf', dpi=300, bbox_inches='tight')
print("Done: ablation_results.pdf")
