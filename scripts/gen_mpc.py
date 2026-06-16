"""MPC comparison figure - academic style, 4 models."""
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

methods = ['LSTM', 'GRU', 'Mamba', 'S4D-WM']
loop_time = [299, 1265, 1296, 1298]  # ms
freq = [3.3, 0.8, 0.8, 0.8]  # Hz
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

# (a) Loop time
bars1 = ax1.bar(methods, loop_time, color=colors, alpha=0.85, edgecolor='white', width=0.55)
ax1.set_ylabel('回路时间 (ms)', fontproperties=zhfont)
ax1.set_title('(a) MPC回路时间', fontproperties=zhfont, fontsize=10, pad=8)
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars1, loop_time):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val}', ha='center', va='bottom', fontsize=8)

# (b) Control frequency
bars2 = ax2.bar(methods, freq, color=colors, alpha=0.85, edgecolor='white', width=0.55)
ax2.set_ylabel('控制频率 (Hz)', fontproperties=zhfont)
ax2.set_title('(b) MPC控制频率', fontproperties=zhfont, fontsize=10, pad=8)
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add 10Hz reference line
ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.text(3.3, 10.3, '典型控制需求 (10Hz)', fontproperties=zhfont, fontsize=7,
         color='gray', ha='right')
for bar, val in zip(bars2, freq):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/mpc_comparison.pdf', dpi=300, bbox_inches='tight')
print("Done: mpc_comparison.pdf")
