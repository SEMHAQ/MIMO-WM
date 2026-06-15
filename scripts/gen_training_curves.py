"""Generate training curves - top-bottom layout, Chinese labels, D4RL data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json, os

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

with open('experiments/d4rl_all_experiments.json') as f:
    data = json.load(f)

logs = data['training_logs']
models = ['S4D-WM_d4rl', 'Mamba-WM_d4rl', 'Trans-WM_d4rl', 'LSTM-WM_d4rl']
labels = ['S4D-WM', 'Mamba-WM', 'Trans.-WM', 'LSTM-WM']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0))

for i, (model, label) in enumerate(zip(models, labels)):
    entries = logs[model]
    epochs = [e['epoch'] for e in entries]
    train_loss = [e['train'] for e in entries]
    val_loss = [e['val'] for e in entries]
    
    ax1.plot(epochs, train_loss, '-', color=colors[i], linewidth=1.5, label=label, alpha=0.8)
    ax2.plot(epochs, val_loss, '-', color=colors[i], linewidth=1.5, label=label, alpha=0.8)

ax1.set_xlabel('训练轮次', fontproperties=zhfont)
ax1.set_ylabel('训练MSE', fontproperties=zhfont)
ax1.legend(fontsize=8.5)
ax1.grid(True, alpha=0.3)
ax1.set_title('(a) 训练损失变化', fontproperties=zhfont, fontsize=10, pad=8)

ax2.set_xlabel('训练轮次', fontproperties=zhfont)
ax2.set_ylabel('验证MSE', fontproperties=zhfont)
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.3)
ax2.set_title('(b) 验证损失变化', fontproperties=zhfont, fontsize=10, pad=8)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/training_curves.pdf', dpi=300, bbox_inches='tight')
print("Done: training_curves.pdf")
