"""MPC comparison - horizontal bars, Nature/Science style, clean typography."""
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

methods = ['S4D-WM', 'Mamba', 'GRU', 'LSTM']
loop_time = [1298, 1296, 1265, 299]
freq = [0.8, 0.8, 0.8, 3.3]

# Color: highlight our model, muted for others
colors = ['#1f77b4', '#B0B0B0', '#B0B0B0', '#B0B0B0']
edge_colors = ['#1565C0', '#888888', '#888888', '#888888']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
y = np.arange(len(methods))
h = 0.5

# ---- (a) Loop time (horizontal) ----
bars1 = ax1.barh(y, loop_time, height=h, color=colors, edgecolor=edge_colors,
                 linewidth=0.8, zorder=3)
ax1.set_xlabel('回路时间 (ms)', fontproperties=zhfont)
ax1.set_yticks(y)
ax1.set_yticklabels(methods, fontsize=9, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.2, axis='x', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, max(loop_time) * 1.25)
for bar, val in zip(bars1, loop_time):
    ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f'{val} ms', ha='left', va='center', fontsize=8, fontweight='bold')
ax1.set_title('(a) MPC回路时间', fontproperties=zhfont, fontsize=10, pad=8)

# ---- (b) Control frequency (horizontal) ----
bars2 = ax2.barh(y, freq, height=h, color=colors, edgecolor=edge_colors,
                 linewidth=0.8, zorder=3)
ax2.set_xlabel('控制频率 (Hz)', fontproperties=zhfont)
ax2.set_yticks(y)
ax2.set_yticklabels(methods, fontsize=9, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.2, axis='x', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axvline(x=10, color='#CC3333', linestyle='--', alpha=0.6, linewidth=1, zorder=2)
ax2.text(10.2, -0.6, '典型控制需求\n(10 Hz)', fontproperties=zhfont_s,
         fontsize=7, color='#CC3333', va='top')
ax2.set_xlim(0, max(freq) * 3.5)
for bar, val in zip(bars2, freq):
    ax2.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
             f'{val} Hz', ha='left', va='center', fontsize=8, fontweight='bold')
ax2.set_title('(b) MPC控制频率', fontproperties=zhfont, fontsize=10, pad=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
