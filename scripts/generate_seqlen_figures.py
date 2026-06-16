"""SeqLen sensitivity: bar+line combo, two recommendation zones per subplot."""
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

T_vals = [16, 32, 64, 128, 256]
x = np.arange(len(T_vals))
w = 0.4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.8), sharex=False)

# ---- Top: MSE bar + R2 line (Humanoid) ----
mse_h = [r['mse'] for r in results['humanoid']]
r2_h = [r['r2'] for r in results['humanoid']]

bars = ax1.bar(x - w/2, mse_h, w, color='#2E86AB', alpha=0.75, edgecolor='white',
               linewidth=0.5, label='Humanoid MSE', zorder=3)
ax1_r = ax1.twinx()
ax1_r.plot(x, r2_h, 'o-', color='#E74C3C', linewidth=1.5, markersize=5,
           label='Humanoid R²', zorder=4)
ax1_r.set_ylabel('R²', fontsize=9)
ax1_r.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)

# Ant data on same top plot
mse_a = [r['mse'] for r in results['ant']]
r2_a = [r['r2'] for r in results['ant']]
bars_a = ax1.bar(x + w/2, mse_a, w, color='#A23B72', alpha=0.55, edgecolor='white',
                 linewidth=0.5, label='Ant MSE', zorder=3)
ax1_r.plot(x, r2_a, 's--', color='#F39C12', linewidth=1.3, markersize=4,
           label='Ant R²', zorder=4)

# Recommendation zones
ax1.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax1.text(0.0, ax1.get_ylim()[1]*0.92 if ax1.get_ylim()[1] > 0 else 1.5,
         'Humanoid\n推荐', fontproperties=zhfont_s, fontsize=7, color='#2E86AB',
         fontweight='bold', va='top')
ax1.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)
ax1.text(3.55, ax1.get_ylim()[1]*0.92 if ax1.get_ylim()[1] > 0 else 1.5,
         'Ant\n推荐', fontproperties=zhfont_s, fontsize=7, color='#A23B72',
         fontweight='bold', va='top')

ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(['16', '32', '64', '128', '256'])
ax1.grid(True, alpha=0.25, axis='y', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.set_title('(a) Humanoid (348维)', fontproperties=zhfont, fontsize=10, pad=8)

# ---- Bottom: MSE bar + R2 line (Ant) ----
bars2 = ax2.bar(x - w/2, mse_a, w, color='#A23B72', alpha=0.75, edgecolor='white',
                linewidth=0.5, label='Ant MSE', zorder=3)
ax2_r = ax2.twinx()
ax2_r.plot(x, r2_a, 's-', color='#F39C12', linewidth=1.5, markersize=5,
           label='Ant R²', zorder=4)
ax2_r.set_ylabel('R²', fontsize=9)
ax2_r.axhline(y=0, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)

# Humanoid data on same bottom plot
bars2_h = ax2.bar(x + w/2, mse_h, w, color='#2E86AB', alpha=0.55, edgecolor='white',
                  linewidth=0.5, label='Humanoid MSE', zorder=3)
ax2_r.plot(x, r2_h, 'o--', color='#E74C3C', linewidth=1.3, markersize=4,
           label='Humanoid R²', zorder=4)

# Recommendation zones
ax2.axvspan(-0.5, 1.5, alpha=0.08, color='#2E86AB', zorder=1)
ax2.text(0.0, ax2.get_ylim()[1]*0.92 if ax2.get_ylim()[1] > 0 else 0.8,
         'Humanoid\n推荐', fontproperties=zhfont_s, fontsize=7, color='#2E86AB',
         fontweight='bold', va='top')
ax2.axvspan(3.5, 4.5, alpha=0.08, color='#A23B72', zorder=1)
ax2.text(3.55, ax2.get_ylim()[1]*0.92 if ax2.get_ylim()[1] > 0 else 0.8,
         'Ant\n推荐', fontproperties=zhfont_s, fontsize=7, color='#A23B72',
         fontweight='bold', va='top')

ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
ax2.set_ylabel('MSE', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['16', '32', '64', '128', '256'])
ax2.grid(True, alpha=0.25, axis='y', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.set_title('(b) Ant (105维)', fontproperties=zhfont, fontsize=10, pad=8)

# Shared legend below both subplots
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2_r.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2,
           loc='lower center', ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True,
           framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Done: seqlen_sensitivity.pdf")
