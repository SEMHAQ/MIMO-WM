"""Generate error accumulation curve figure (E4).
S4D-WM vs Mamba-WM: MSE vs rollout step H."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = '/mnt/c/Windows/Fonts/simhei.ttf'
zh_font = FontProperties(fname=font_path, size=10)
zh_font_sm = FontProperties(fname=font_path, size=8.5)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

with open('experiments/error_accumulation.json') as f:
    data = json.load(f)

H_VALUES = [1, 2, 4, 8, 16]
colors = {'S4D-WM': '#2196F3', 'Mamba-WM': '#FF5722'}

fig, ax = plt.subplots(figsize=(4.5, 3.2))

for model_name in ['S4D-WM', 'Mamba-WM']:
    means = []
    stds = []
    for H in H_VALUES:
        key = f'H={H}'
        vals = [r['mse_avg'] for r in data[model_name][key]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    
    means = np.array(means)
    stds = np.array(stds)
    
    ax.plot(H_VALUES, means, 'o-', color=colors[model_name], 
            label=model_name, linewidth=1.5, markersize=5)
    ax.fill_between(H_VALUES, means - stds, means + stds, 
                     alpha=0.15, color=colors[model_name])

ax.set_xlabel('Rollout Horizon H', fontsize=10)
ax.set_ylabel('MSE', fontsize=10)
ax.set_xticks(H_VALUES)
ax.set_xticklabels([str(h) for h in H_VALUES])
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)

# Add crossover annotation
ax.annotate('S4D优势区', xy=(4, 0.24), fontsize=8, fontproperties=zh_font_sm,
            color='#2196F3', ha='center')
ax.annotate('Mamba反超', xy=(16, 0.66), fontsize=8, fontproperties=zh_font_sm,
            color='#FF5722', ha='center', va='top')

plt.tight_layout()
plt.savefig('paper/figures/error_accumulation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/error_accumulation.png', dpi=300, bbox_inches='tight')
print("Saved: paper/figures/error_accumulation.pdf")
