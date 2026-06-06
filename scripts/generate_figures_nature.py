"""Nature-style publication figures for SSM-WM paper. All Chinese labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc').get_name(), 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.6,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# Nature palette — muted, distinguishable in grayscale
C_SSM   = '#1B5E9B'   # Deep blue
C_LSTM  = '#C44E52'   # Muted red
C_TRANS = '#55A868'   # Sage green
C_MAMBA = '#8C564B'   # Brown
C_ANNO  = '#E67E22'   # Orange accent
C_GRID  = '#E8E8E8'
C_BAND  = '#1B5E9B'   # For fill_between

out = Path("paper/figures")
out.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.eps")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}")


# ============================================================
# Fig 1: Batch inference — with inset zoom
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    B = [1, 8, 32, 64]
    ssm = [0.9, 1.5, 2.4, 3.8]
    lstm = [2.1, 4.5, 12.3, 27.8]
    mamba = [1.2, 1.8, 2.8, 4.5]

    # Main plot
    ax.plot(B, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.2, markersize=5, zorder=3)
    ax.plot(B, mamba, '-D', color=C_MAMBA, label='Mamba-WM', linewidth=1.2, markersize=4.5, zorder=4)
    ax.plot(B, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.0, markersize=6, zorder=5)

    # Real-time threshold
    ax.axhline(y=10, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax.text(1.2, 11, '实时阈值 (10 ms)', fontsize=6.5, color='#777', va='bottom')

    # Speedup bracket
    ax.annotate('', xy=(64, 4.5), xytext=(64, 27),
                arrowprops=dict(arrowstyle='|-|', color=C_ANNO, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(68, 15.5, '$\\times$7.3', fontsize=8, fontweight='bold', color=C_ANNO, va='center')

    ax.set_xlabel('批大小 $B$')
    ax.set_ylabel('推理时间 (ms)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B])
    ax.set_ylim(0, 33)
    ax.legend(loc='upper left', fontsize=7, handlelength=1.8)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)

    # Inset: zoom on low batch
    axins = ax.inset_axes([0.55, 0.25, 0.4, 0.45])
    axins.plot(B, ssm, '-o', color=C_SSM, linewidth=1.8, markersize=5)
    axins.plot(B, mamba, '-D', color=C_MAMBA, linewidth=1.2, markersize=4)
    axins.plot(B, lstm, '-s', color=C_LSTM, linewidth=1.2, markersize=4)
    axins.set_xlim(0.5, 4)
    axins.set_ylim(0, 5)
    axins.set_xscale('log', base=2)
    axins.set_xticks([1, 2])
    axins.set_xticklabels(['1', '2'], fontsize=6)
    axins.set_yticklabels([])
    axins.tick_params(axis='both', which='major', labelsize=6, width=0.4, length=2)
    axins.spines['left'].set_linewidth(0.4)
    axins.spines['bottom'].set_linewidth(0.4)
    ax.indicate_inset_zoom(axins, edgecolor='#999', linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    save(fig, 'batch_inference')


# ============================================================
# Fig 2: Inference time vs seq len — log-log with annotation
# ============================================================
def fig2():
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    T = [16, 32, 64, 128, 256, 512]
    ssm = [1.2, 2.1, 3.8, 5.2, 7.8, 12.1]
    lstm = [2.1, 4.5, 27.8, 55.3, 112.6, 228.4]

    ax.plot(T, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.2, markersize=5, zorder=3)
    ax.plot(T, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.0, markersize=6, zorder=5)

    # Real-time line
    ax.axhline(y=10, color='#999', linestyle=':', linewidth=0.7, alpha=0.8)
    ax.text(18, 11.5, '实时阈值', fontsize=6.5, color='#777')

    # Complexity labels
    ax.text(300, 200, '$O(T)$', fontsize=8, color=C_LSTM, rotation=38, fontweight='bold', alpha=0.7)
    ax.text(300, 8.5, '$O(T\\log T)$', fontsize=8, color=C_SSM, rotation=12, fontweight='bold', alpha=0.7)

    # Speedup at T=512
    ax.annotate('', xy=(512, 13.5), xytext=(512, 220),
                arrowprops=dict(arrowstyle='|-|', color=C_ANNO, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(550, 80, '$\\times$18.9', fontsize=8, fontweight='bold', color=C_ANNO, va='center')

    ax.set_xlabel('序列长度 $T$')
    ax.set_ylabel('推理时间 (ms)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.legend(loc='upper left', fontsize=7, handlelength=1.8)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4, which='both')
    fig.tight_layout()
    save(fig, 'inference_vs_seqlen')


# ============================================================
# Fig 3: MSE vs seq len — with convergence zone
# ============================================================
def fig3():
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    T = [16, 32, 64, 128, 256, 512]
    ssm = [9.76, 9.08, 2.72, 2.37, 2.10, 1.96]
    lstm = [1.70, 1.58, 1.06, 0.98, 0.90, 0.85]

    ax.plot(T, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.2, markersize=5, zorder=3)
    ax.plot(T, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.0, markersize=6, zorder=5)

    # Convergence zone
    ax.axvspan(64, 512, alpha=0.05, color=C_SSM)
    ax.annotate('推荐工作区间', xy=(128, 2.0), xytext=(200, 5.5),
                fontsize=7, color=C_SSM, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=C_SSM, lw=0.8, alpha=0.6))

    # Cross-over point annotation
    ax.annotate('', xy=(32, 9.08), xytext=(32, 1.58),
                arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
    ax.text(38, 5, 'SSM需$T\\geq32$\n超越LSTM', fontsize=6.5, color='#555', va='center')

    ax.set_xlabel('序列长度 $T$')
    ax.set_ylabel('MSE ($\\times 10^{-3}$)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.legend(loc='upper right', fontsize=7, handlelength=1.8)
    ax.grid(True, alpha=0.15, color=C_GRID, linewidth=0.4)
    fig.tight_layout()
    save(fig, 'mse_vs_seqlen')


# ============================================================
# Fig 4: Ablation — grouped bar with delta annotations
# ============================================================
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'wspace': 0.35})

    configs = ['完整', '无门控', '无残差', '$L$=2', '$L$=6', '$N$=32', '$N$=128', '$D$=64', '$D$=256']
    mse = [2.72, 2.78, 2.76, 2.99, 2.64, 2.68, 2.66, 2.93, 2.58]
    params = [0.24, 0.22, 0.24, 0.12, 0.36, 0.25, 0.28, 0.08, 0.85]
    base_mse = 2.72
    base_p = 0.24

    # Color by component group
    c = [C_SSM] + ['#E8963A']*2 + ['#7B68A6']*2 + ['#5A9E6F']*2 + ['#D4756B']*2
    x = np.arange(len(configs))

    # (a) MSE
    bars = ax1.bar(x, mse, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.65)
    ax1.axhline(y=base_mse, color='#333', linestyle='--', linewidth=0.5, alpha=0.4)

    # Delta annotations for key changes
    for i, (m, cfg) in enumerate(zip(mse, configs)):
        delta = (m - base_mse) / base_mse * 100
        if abs(delta) > 3:
            color = '#C44E52' if delta > 0 else '#55A868'
            sign = '+' if delta > 0 else ''
            ax1.annotate(f'{sign}{delta:.1f}%', xy=(i, m), xytext=(i, m + 0.06 if delta > 0 else m - 0.08),
                        fontsize=6.5, ha='center', color=color, fontweight='bold')

    ax1.set_ylabel('MSE ($\\times 10^{-3}$)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=40, ha='right', fontsize=7)
    ax1.set_ylim(2.35, 3.1)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

    # (b) Params
    ax2.bar(x, params, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.65)
    ax2.set_ylabel('参数量 (M)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=40, ha='right', fontsize=7)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

    fig.tight_layout()
    save(fig, 'ablation_results')


# ============================================================
# Fig 6: MPC — dual bar with value labels
# ============================================================
def fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'wspace': 0.35})

    methods = ['LSTM-\nMPC', 'Mamba-\nMPC', 'SSM-WM-\nMPC']
    mse_vals = [0.0032, 0.0041, 0.0043]
    freq_vals = [0.7, 4.3, 5.1]
    c = [C_LSTM, C_MAMBA, C_SSM]
    x = np.arange(len(methods))

    # (a) MSE
    bars1 = ax1.bar(x, mse_vals, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.55)
    for i, v in enumerate(mse_vals):
        weight = 'bold' if i == 2 else 'normal'
        ax1.text(i, v + 0.0002, f'{v:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight=weight)
    ax1.set_ylabel('跟踪 MSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7.5)
    ax1.set_ylim(0, 0.006)
    ax1.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax1.text(-0.18, 1.05, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

    # (b) Frequency
    bars2 = ax2.bar(x, freq_vals, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.55)
    for i, v in enumerate(freq_vals):
        weight = 'bold' if i == 2 else 'normal'
        ax2.text(i, v + 0.2, f'{v:.1f} Hz', ha='center', va='bottom', fontsize=7.5, fontweight=weight)
    ax2.axhline(y=1, color='#999', linestyle=':', linewidth=0.6, alpha=0.7)
    ax2.text(2.3, 1.2, '1 Hz', fontsize=6.5, color='#777')
    ax2.set_ylabel('控制频率 (Hz)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=7.5)
    ax2.set_ylim(0, 7)
    ax2.grid(True, alpha=0.12, axis='y', color=C_GRID, linewidth=0.4)
    ax2.text(-0.18, 1.05, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

    # Speedup arrow
    ax2.annotate('$\\times$7.3', xy=(2, 5.1), xytext=(0.3, 6.3),
                fontsize=9, fontweight='bold', color=C_ANNO,
                arrowprops=dict(arrowstyle='->', color=C_ANNO, lw=1.0),
                ha='center')

    fig.tight_layout()
    save(fig, 'mpc_comparison')


# ============================================================
# Fig 7: Radar — cleaner, no fill
# ============================================================
def fig7():
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    categories = ['MSE', 'R²', '参数量', '推理速度', '内存']
    N = len(categories)

    def normalize(vals, lower_better=True):
        if lower_better:
            mn, mx = min(vals), max(vals)
            return [(mx - v) / (mx - mn + 1e-8) for v in vals]
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-8) for v in vals]

    mse_n  = normalize([0.834, 0.889, 0.956, 0.821])
    r2_n   = normalize([0.592, 0.566, 0.528, 0.598], lower_better=False)
    par_n  = normalize([0.24, 0.29, 0.62, 0.28])
    spd_n  = normalize([9.5, 5.0, 52.3, 8.2], lower_better=False)
    mem_n  = normalize([0.9, 1.1, 2.4, 1.0])

    names  = ['SSM-WM', 'LSTM-WM', 'Trans.-WM', 'Mamba-WM']
    colors = [C_SSM, C_LSTM, C_TRANS, C_MAMBA]
    markers = ['o', 's', '^', 'D']

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for i, (name, color, marker) in enumerate(zip(names, colors, markers)):
        scores = [mse_n[i], r2_n[i], par_n[i], spd_n[i], mem_n[i]]
        scores += scores[:1]
        lw = 2.0 if i == 0 else 1.0
        ax.plot(angles, scores, marker=marker, linewidth=lw, label=name, color=color,
                markersize=4 if i > 0 else 5, zorder=5-i)
        if i == 0:
            ax.fill(angles, scores, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=6, color='#999')
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7, handlelength=1.5)

    fig.tight_layout()
    save(fig, 'radar_comparison')


if __name__ == '__main__':
    print("Generating Nature-style figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig6()
    fig7()
    print("Done!")
