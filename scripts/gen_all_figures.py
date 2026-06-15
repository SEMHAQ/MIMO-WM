"""Fix all 4 figures based on user feedback."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os, json

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=11)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'font.size': 10,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

C_SSM = '#1B5E9B'; C_LSTM = '#C44E52'; C_TRANS = '#55A868'; C_MAMBA = '#8C564B'
C_ANNO = '#E67E22'; C_GRID = '#E8E8E8'
OUTDIR = 'paper/figures'

def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'))
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'), dpi=300)
    plt.close(fig)
    print(f'  Saved {name}')

# ============================================================
# Figure 1: Batch inference - line + inset zoom
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    B = [1, 8, 32, 64]
    lstm = [2.9, 1.9, 3.5, 4.5]
    trans = [2.9, 3.1, 3.4, 4.3]
    mamba = [9.5, 10.5, 9.6, 8.3]
    s4d = [8.3, 8.7, 8.8, 8.7]

    ax.plot(B, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.5, markersize=7, zorder=3)
    ax.plot(B, trans, '-^', color=C_TRANS, label='Transformer-WM', linewidth=1.5, markersize=7, zorder=3)
    ax.plot(B, mamba, '-D', color=C_MAMBA, label='Mamba-WM', linewidth=1.5, markersize=6, zorder=4)
    ax.plot(B, s4d, '-o', color=C_SSM, label='S4D-WM', linewidth=2.5, markersize=8, zorder=5)

    ax.axhline(y=10, color='#999', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.text(1.2, 10.8, '实时控制阈值 (10ms)', fontproperties=zhfont_s, fontsize=8, color='#777', va='bottom')

    ax.set_xlabel('批大小 B', fontproperties=zhfont)
    ax.set_ylabel('推理时间 (ms)', fontproperties=zhfont)
    ax.set_xscale('log', base=2)
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B], fontsize=10)
    ax.set_ylim(0, 14)
    ax.legend(loc='upper left', fontsize=8.5, handlelength=1.8, ncol=2)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)

    # Inset zoom for B=1 region
    axins = ax.inset_axes([0.55, 0.45, 0.35, 0.4])
    axins.plot(B, lstm, '-s', color=C_LSTM, linewidth=1.2, markersize=5)
    axins.plot(B, trans, '-^', color=C_TRANS, linewidth=1.2, markersize=5)
    axins.plot(B, mamba, '-D', color=C_MAMBA, linewidth=1.2, markersize=4)
    axins.plot(B, s4d, '-o', color=C_SSM, linewidth=1.8, markersize=6)
    axins.set_xlim(0.5, 2)
    axins.set_ylim(0, 4)
    axins.set_xscale('log', base=2)
    axins.set_xticks([1])
    axins.set_xticklabels(['1'], fontsize=8)
    axins.tick_params(axis='both', which='major', labelsize=8, width=0.4, length=2)
    ax.indicate_inset_zoom(axins, edgecolor='#999', linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    save(fig, 'batch_inference')

# ============================================================
# Figure 2: Seqlen - legend below x-axis
# ============================================================
def fig2():
    T = [16, 32, 64, 128, 256]
    x = np.arange(len(T))
    h_mse = [0.291, 0.442, 0.612, 1.213, 2.146]
    h_r2 = [0.656, 0.479, 0.153, -0.623, -1.694]
    a_mse = [0.542, 0.728, 0.942, 0.934, 0.480]
    a_r2 = [0.302, 0.150, -0.019, 0.139, 0.131]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.5), gridspec_kw={'hspace': 0.35})
    w = 0.3

    # (a) MSE
    ax1.bar(x - w/2, h_mse, w, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax1.bar(x + w/2, a_mse, w, color=C_LSTM, alpha=0.3, edgecolor=C_LSTM, linewidth=0.8, zorder=2)
    ax1.plot(x - w/2, h_mse, '-o', color=C_SSM, linewidth=2.0, markersize=6, zorder=5, label='Humanoid (348D)')
    ax1.plot(x + w/2, a_mse, '-s', color=C_LSTM, linewidth=2.0, markersize=6, zorder=5, label='Ant (105D)')
    ax1.axvspan(-0.3, 1.3, alpha=0.06, color=C_SSM, zorder=0)
    ax1.axvspan(2.7, 4.3, alpha=0.06, color=C_LSTM, zorder=0)
    ax1.annotate('Humanoid推荐', xy=(0.5, 1.9), fontproperties=zhfont_s, fontsize=8, color=C_SSM, ha='center', fontstyle='italic')
    ax1.annotate('Ant推荐', xy=(4, 1.9), fontproperties=zhfont_s, fontsize=8, color=C_LSTM, ha='center', fontstyle='italic')
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in T], fontsize=10)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    ax1.legend(fontsize=9, handlelength=1.8, loc='upper left')

    # (b) R²
    ax2.bar(x - w/2, h_r2, w, color=C_SSM, alpha=0.3, edgecolor=C_SSM, linewidth=0.8, zorder=2)
    ax2.bar(x + w/2, a_r2, w, color=C_LSTM, alpha=0.3, edgecolor=C_LSTM, linewidth=0.8, zorder=2)
    ax2.plot(x - w/2, h_r2, '-o', color=C_SSM, linewidth=2.0, markersize=6, zorder=5, label='Humanoid (348D)')
    ax2.plot(x + w/2, a_r2, '-s', color=C_LSTM, linewidth=2.0, markersize=6, zorder=5, label='Ant (105D)')
    ax2.axvspan(-0.3, 1.3, alpha=0.06, color=C_SSM, zorder=0)
    ax2.axvspan(2.7, 4.3, alpha=0.06, color=C_LSTM, zorder=0)
    ax2.axhline(y=0, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax2.set_xlabel('序列长度 T', fontproperties=zhfont)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in T], fontsize=10)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    ax2.legend(fontsize=9, handlelength=1.8, loc='lower left')

    fig.subplots_adjust(bottom=0.12)
    save(fig, 'seqlen_sensitivity')

# ============================================================
# Figure 4: Training curves - all 4 models' validation curves
# ============================================================
def fig4():
    with open('experiments/d4rl_all_experiments.json') as f:
        data = json.load(f)
    logs = data['training_logs']

    models = ['S4D-WM_d4rl', 'Mamba-WM_d4rl', 'Trans-WM_d4rl', 'LSTM-WM_d4rl']
    labels = ['S4D-WM', 'Mamba-WM', 'Transformer-WM', 'LSTM-WM']
    colors = [C_SSM, C_MAMBA, C_TRANS, C_LSTM]
    lws = [2.5, 1.5, 1.5, 1.5]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for model, label, color, lw in zip(models, labels, colors, lws):
        entries = logs[model]
        epochs = [e['epoch'] for e in entries]
        val_mse = [e['val'] for e in entries]
        ax.plot(epochs, val_mse, '-', color=color, linewidth=lw, label=label, alpha=0.85)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('验证MSE', fontproperties=zhfont, fontsize=11)
    ax.legend(fontsize=9, handlelength=1.8)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    ax.set_xlim(0, 100)

    # Annotate best values
    for model, color in zip(models, [C_SSM, C_MAMBA, C_TRANS, C_LSTM]):
        entries = logs[model]
        val_mse = [e['val'] for e in entries]
        best_ep = np.argmin(val_mse)
        best_val = min(val_mse)
        ax.plot(best_ep, best_val, 'o', color=color, markersize=6, zorder=5)

    fig.tight_layout()
    save(fig, 'training_curves')

# ============================================================
# Figure 5: MPC - top-bottom with gradient bars
# ============================================================
def fig5():
    methods = ['LSTM-MPC', 'Mamba-MPC', 'S4D-WM-MPC']
    loop_ms = [299, 1296, 1298]
    freq_hz = [3.3, 0.8, 0.8]
    colors = [C_LSTM, C_MAMBA, C_SSM]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.0))
    y = np.arange(len(methods))

    # (a) Loop time - horizontal bars with gradient effect
    bars1 = ax1.barh(y, loop_ms, height=0.5, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars1, loop_ms):
        ax1.text(v + 25, bar.get_y() + bar.get_height()/2,
                f'{v} ms', fontsize=10, va='center', fontweight='bold', color='#333')
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('控制回路时间 (ms)', fontproperties=zhfont, fontsize=11)
    ax1.set_title('(a) 控制回路时间', fontproperties=zhfont, fontsize=11, pad=8)
    ax1.set_xlim(0, 1600)

    # (b) Frequency
    bars2 = ax2.barh(y, freq_hz, height=0.5, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars2, freq_hz):
        ax2.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f'{v:.1f} Hz', fontsize=10, va='center', fontweight='bold', color='#333')
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('控制频率 (Hz)', fontproperties=zhfont, fontsize=11)
    ax2.set_title('(b) 控制频率', fontproperties=zhfont, fontsize=11, pad=8)
    ax2.set_xlim(0, 4.5)

    # Speedup annotation
    ax1.annotate('', xy=(299, -0.35), xytext=(1298, -0.35),
                arrowprops=dict(arrowstyle='<->', color=C_ANNO, lw=1.5))
    ax1.text(798, -0.55, '4.3×更快', fontproperties=zhfont, fontsize=9, ha='center', color=C_ANNO, fontweight='bold')

    fig.tight_layout()
    save(fig, 'mpc_comparison')

# Run
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    fig1()
    fig2()
    fig4()
    fig5()
    print("All figures done!")
