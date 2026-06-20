"""MPC comparison - gradient vs CEM, Nature/Science style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

methods = ['LSTM', 'GRU', 'Mamba', 'S4D-WM']
# Gradient MPC (Table 10)
grad_time = [1356, 1307, 5779, 5768]
grad_freq = [0.7, 0.8, 0.2, 0.2]
# CEM-MPC (Table 11)
cem_time = [32, 19, 78, 76]
cem_freq = [31.3, 51.9, 12.9, 13.1]

colors_grad = ['#D4D4D4', '#D4D4D4', '#D4D4D4', '#A0A0A0']
colors_cem = ['#87CEEB', '#87CEEB', '#87CEEB', '#1f77b4']

x = np.arange(len(methods))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4.5))

# (a) Loop time - grouped bars
bars1 = ax1.bar(x - width/2, grad_time, width, color=colors_grad, edgecolor='gray', linewidth=0.5, label='梯度MPC', zorder=3)
bars2 = ax1.bar(x + width/2, cem_time, width, color=colors_cem, edgecolor='gray', linewidth=0.5, label='CEM-MPC', zorder=3)
ax1.set_ylabel('回路时间 (ms)', fontproperties=zhfont)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9)
ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(fontsize=8, loc='upper left')
# Only label CEM bars (more relevant)
for bar, val in zip(bars2, cem_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax1.set_title('(a) MPC回路时间', fontproperties=zhfont, fontsize=10, pad=8)

# (b) Control frequency - grouped bars
bars3 = ax2.bar(x - width/2, grad_freq, width, color=colors_grad, edgecolor='gray', linewidth=0.5, label='梯度MPC', zorder=3)
bars4 = ax2.bar(x + width/2, cem_freq, width, color=colors_cem, edgecolor='gray', linewidth=0.5, label='CEM-MPC', zorder=3)
ax2.set_ylabel('控制频率 (Hz)', fontproperties=zhfont)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=10, color='#CC3333', linestyle='--', alpha=0.6, linewidth=1, zorder=2)
ax2.text(len(methods)-0.5, 10.8, '10Hz需求', fontproperties=zhfont_s,
         fontsize=7.5, color='#CC3333', ha='right')
ax2.legend(fontsize=8, loc='upper left')
for bar, val in zip(bars4, cem_freq):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax2.set_title('(b) MPC控制频率', fontproperties=zhfont, fontsize=10, pad=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
