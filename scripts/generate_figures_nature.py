"""Publication-quality figures for SSM-WM paper (nature style)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Chinese font
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc').get_name(), 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 9,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

# Nature-style color palette
C_SSM   = '#0F4D92'   # Deep blue (primary)
C_LSTM  = '#B64342'   # Muted red
C_TRANS = '#42949E'   # Teal
C_MAMBA = '#9A4D8E'   # Violet
C_GRID  = '#E0E0E0'   # Light grid
C_ANNO  = '#E67E22'   # Orange for annotations

out = Path("paper/figures")
out.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.eps")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}")


# === Fig 1: Batch inference ===
def fig1():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    B = [1, 8, 32, 64]
    ssm = [0.9, 1.5, 2.4, 3.8]
    lstm = [2.1, 4.5, 12.3, 27.8]
    mamba = [1.2, 1.8, 2.8, 4.5]

    ax.plot(B, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.2, markersize=7, zorder=5)
    ax.plot(B, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.5, markersize=6)
    ax.plot(B, mamba, '-D', color=C_MAMBA, label='Mamba-WM', linewidth=1.5, markersize=5)
    ax.axhline(y=10, color='#CC0000', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(65, 10.5, '10 ms (实时阈值)', fontsize=7.5, color='#CC0000', ha='right', va='bottom')

    # Speedup annotation
    ax.annotate('', xy=(64, 3.8), xytext=(64, 27.8),
                arrowprops=dict(arrowstyle='<->', color=C_ANNO, lw=1.5))
    ax.text(67, 15, '7.3×', fontsize=9, fontweight='bold', color=C_ANNO, va='center')

    ax.set_xlabel('批大小 $B$')
    ax.set_ylabel('推理时间 (ms)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B])
    ax.set_ylim(0, 35)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2, color=C_GRID)
    fig.tight_layout()
    save(fig, 'batch_inference')


# === Fig 2: Inference time vs seq len ===
def fig2():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    T = [16, 32, 64, 128, 256, 512]
    ssm = [1.2, 2.1, 3.8, 5.2, 7.8, 12.1]
    lstm = [2.1, 4.5, 27.8, 55.3, 112.6, 228.4]

    ax.plot(T, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.2, markersize=7, zorder=5)
    ax.plot(T, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.5, markersize=6)
    ax.axhline(y=10, color='#CC0000', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(520, 10.5, '10 ms', fontsize=7.5, color='#CC0000', ha='right', va='bottom')

    # Speedup at T=512
    ax.annotate('', xy=(512, 12.1), xytext=(512, 228.4),
                arrowprops=dict(arrowstyle='<->', color=C_ANNO, lw=1.5))
    ax.text(540, 100, '18.9×', fontsize=9, fontweight='bold', color=C_ANNO, va='center')

    ax.set_xlabel('序列长度 $T$')
    ax.set_ylabel('推理时间 (ms)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2, color=C_GRID, which='both')
    fig.tight_layout()
    save(fig, 'inference_vs_seqlen')


# === Fig 3: MSE vs seq len ===
def fig3():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    T = [16, 32, 64, 128, 256, 512]
    ssm = [9.76, 9.08, 2.72, 2.37, 2.10, 1.96]
    lstm = [1.70, 1.58, 1.06, 0.98, 0.90, 0.85]

    ax.plot(T, ssm, '-o', color=C_SSM, label='SSM-WM', linewidth=2.2, markersize=7, zorder=5)
    ax.plot(T, lstm, '-s', color=C_LSTM, label='LSTM-WM', linewidth=1.5, markersize=6)

    # Convergence zone
    ax.axvspan(64, 512, alpha=0.06, color=C_SSM)
    ax.text(128, 9.5, '推荐工作区间\n$T \\geq 64$', fontsize=7.5, color=C_SSM, ha='center', va='top',
            style='italic')

    ax.set_xlabel('序列长度 $T$')
    ax.set_ylabel('MSE ($\\times 10^{-3}$)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, color=C_GRID)
    fig.tight_layout()
    save(fig, 'mse_vs_seqlen')


# === Fig 4: Ablation ===
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) MSE
    configs = ['完整', '无门控', '无残差', '$L$=2', '$L$=6', '$N$=32', '$N$=128', '$D$=64', '$D$=256']
    mse = [2.72, 2.78, 2.76, 2.99, 2.64, 2.68, 2.66, 2.93, 2.58]
    base_mse = 2.72

    bar_colors = [C_SSM] + ['#FF9800']*2 + ['#9C27B0']*2 + ['#4CAF50']*2 + ['#E67E22']*2
    x = np.arange(len(configs))
    bars = ax1.bar(x, mse, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.7)
    ax1.axhline(y=base_mse, color='#333', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.text(8.3, base_mse + 0.02, 'Baseline', fontsize=7, color='#333', ha='right', va='bottom')

    # Highlight key differences
    for i, (m, c) in enumerate(zip(mse, configs)):
        if c in ['无门控', '$D$=64']:
            ax1.annotate(f'+{(m-base_mse)/base_mse*100:.1f}%', xy=(i, m), xytext=(i, m+0.08),
                        fontsize=7, ha='center', color='#CC0000', fontweight='bold')

    ax1.set_ylabel('MSE ($\\times 10^{-3}$)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=7.5)
    ax1.set_ylim(2.4, 3.15)
    ax1.grid(True, alpha=0.2, axis='y', color=C_GRID)

    # (b) Params
    params = [0.24, 0.22, 0.24, 0.12, 0.36, 0.25, 0.28, 0.08, 0.85]
    bars2 = ax2.bar(x, params, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.7)
    ax2.set_ylabel('参数量 (M)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=7.5)
    ax2.grid(True, alpha=0.2, axis='y', color=C_GRID)

    fig.tight_layout()
    save(fig, 'ablation_results')


# === Fig 6: MPC comparison ===
def fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    methods = ['LSTM-MPC', 'Mamba-MPC', 'SSM-WM-MPC']
    mse_vals = [0.0032, 0.0041, 0.0043]
    freq_vals = [0.7, 4.3, 5.1]
    bar_colors = [C_LSTM, C_MAMBA, C_SSM]

    x = np.arange(len(methods))

    # (a) MSE
    bars1 = ax1.bar(x, mse_vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.6)
    for i, v in enumerate(mse_vals):
        ax1.text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_ylabel('跟踪 MSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylim(0, 0.006)
    ax1.grid(True, alpha=0.2, axis='y', color=C_GRID)
    ax1.set_title('(a) 跟踪精度', fontsize=10, fontweight='bold')

    # (b) Frequency
    bars2 = ax2.bar(x, freq_vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8, width=0.6)
    for i, v in enumerate(freq_vals):
        ax2.text(i, v + 0.15, f'{v:.1f} Hz', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.axhline(y=1, color='#CC0000', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.text(2.3, 1.1, '1 Hz (最低需求)', fontsize=7, color='#CC0000', ha='right')
    ax2.set_ylabel('控制频率 (Hz)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylim(0, 7)
    ax2.grid(True, alpha=0.2, axis='y', color=C_GRID)
    ax2.set_title('(b) 控制频率', fontsize=10, fontweight='bold')

    # Speedup annotation
    ax2.annotate('7.3×', xy=(2, 5.1), xytext=(0.5, 6.2),
                fontsize=10, fontweight='bold', color=C_ANNO,
                arrowprops=dict(arrowstyle='->', color=C_ANNO, lw=1.2))

    fig.tight_layout()
    save(fig, 'mpc_comparison')


if __name__ == '__main__':
    print("Generating nature-style figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig6()
    print("Done!")
