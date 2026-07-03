"""生成论文图3(消融实验)和图4(序列长度敏感性)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=7)



plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'mathtext.fontset': 'stix',
})


# ============================================================
def gen_ablation():
    configs = [
        ('MIMO-WM',       18.69, 0.31, 0.766, 0.208),
        ('w/o gate',      20.36, 0.10, 0.745, 0.142),
        ('w/o residual',  26.14, 0.26, 0.673, 0.208),
        ('w/o LayerNorm', 21.25, 0.48, 0.734, 0.208),
        ('SSM$\\to$LSTM', 38.10, 0.44, 0.523, 0.389),
        ('SSM$\\to$GRU',  32.94, 0.58, 0.588, 0.323),
        ('$D$=64',         21.69, 0.35, 0.729, 0.080),
        ('$D$=256',        18.15, 0.26, 0.773, 0.613),
        ('$L$=1',          19.48, 0.21, 0.756, 0.166),
        ('$L$=4',          18.49, 0.22, 0.769, 0.292),
        ('$N$=8',          18.77, 0.22, 0.765, 0.200),
    ]

    n = len(configs)
    labels = [c[0] for c in configs]
    mses = [c[1] for c in configs]
    mse_stds = [c[2] for c in configs]
    r2s = [c[3] for c in configs]
    params = [c[4] for c in configs]

    BAR = '#4A90D9'
    HL = '#E74C3C'
    colors = [HL if i == 0 else BAR for i in range(n)]

    fig = plt.figure(figsize=(7.0, 8.5))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.12,
                          left=0.16, right=0.96, top=0.95, bottom=0.06)
    ax_mse = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1], sharey=ax_mse)
    ax_params = fig.add_subplot(gs[1, :])

    y = np.arange(n)
    h = 0.6

    # (a) MSE
    ax_mse.barh(y, mses, height=h, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    ax_mse.errorbar(mses, y, xerr=mse_stds, fmt='none', ecolor='#333', capsize=2, linewidth=0.8, zorder=4)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_mse.text(mses[i] + mse_stds[i] + 0.5, y[i], f'{mses[i]:.1f}',
                    fontsize=8, va='center', color='#222', fontweight=fw)
    ax_mse.set_xlabel('MSE ($\\times 10^{-2}$)', fontsize=11)
    ax_mse.set_xlim(0, 44)
    ax_mse.set_title('(a) MSE', fontsize=11, fontweight='bold', loc='left')
    ax_mse.spines['top'].set_visible(False)
    ax_mse.spines['right'].set_visible(False)
    ax_mse.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_mse.tick_params(axis='x', labelsize=9)

    # (b) R2
    ax_r2.barh(y, r2s, height=h, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_r2.text(r2s[i] + 0.003, y[i], f'{r2s[i]:.3f}',
                   fontsize=8, va='center', color='#222', fontweight=fw)
    ax_r2.set_xlabel('$R^2$', fontsize=11)
    ax_r2.set_xlim(0.48, 0.80)
    ax_r2.set_title('(b) $R^2$', fontsize=11, fontweight='bold', loc='left')
    ax_r2.spines['top'].set_visible(False)
    ax_r2.spines['right'].set_visible(False)
    ax_r2.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_r2.tick_params(axis='x', labelsize=9)
    plt.setp(ax_r2.get_yticklabels(), visible=False)

    # (c) Parameters
    bubble_sizes = [p * 500 for p in params]
    ax_params.scatter(params, y, s=bubble_sizes, c=colors, edgecolors='white',
                      linewidth=0.8, zorder=3, alpha=0.85)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_params.text(params[i] + 0.015, y[i], f'{params[i]:.3f}',
                       fontsize=8, va='center', color='#222', fontweight=fw)
    ax_params.set_xlabel('Parameters (M)', fontsize=11)
    ax_params.set_xlim(0, 0.75)
    ax_params.set_title('(c) Parameters', fontsize=11, fontweight='bold', loc='left')
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_visible(False)
    ax_params.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_params.tick_params(axis='x', labelsize=9)

    # Y轴: 统一顺序
    ax_mse.invert_yaxis()
    ax_params.set_ylim(ax_mse.get_ylim())

    ax_mse.set_yticks(y)
    ax_mse.set_yticklabels(labels, fontsize=9)
    for i, tick in enumerate(ax_mse.get_yticklabels()):
        if i == 0:
            tick.set_fontweight('bold')
            tick.set_color(HL)

    ax_params.set_yticks(y)
    ax_params.set_yticklabels(labels, fontsize=9)
    for i, tick in enumerate(ax_params.get_yticklabels()):
        if i == 0:
            tick.set_fontweight('bold')
            tick.set_color(HL)

    plt.savefig('paper/figures/ablation_results.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('paper/figures/ablation_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print('Done: ablation_results.pdf')

# ============================================================
def gen_seqlen():
    ts = np.array([8, 16, 32, 64, 128])

    # Humanoid
    h_mse = [20.14, 19.23, 21.18, 21.28, 41.13]
    h_mse_std = [0.13, 0.14, 0.04, 0.16, 0.36]
    h_r2 = [0.765, 0.764, 0.735, 0.708, 0.448]

    # HumanoidStandup (placeholder, replace after experiment)
    hs_mse = [49.66, 48.5, 53.10, 55.0, 75.0]
    hs_r2 = [0.480, 0.490, 0.444, 0.420, 0.300]

    c_h = '#3498db'
    c_hs = '#e67e22'
    w = 2.5

    fig, (ax_mse, ax_r2) = plt.subplots(2, 1, figsize=(5.5, 6.0), sharex=True)
    fig.patch.set_facecolor('white')

    # === (a) MSE ===
    ax_mse.bar(ts - w/2 - 0.3, h_mse, width=w, color=c_h, alpha=0.8,
               edgecolor='white', linewidth=0.5, label='Humanoid')
    ax_mse.bar(ts + w/2 + 0.3, hs_mse, width=w, color=c_hs, alpha=0.8,
               edgecolor='white', linewidth=0.5, label='HumanoidStandup')
    ax_mse.plot(ts - w/2 - 0.3, h_mse, 'o--', color=c_h, markersize=4, linewidth=1.2)
    ax_mse.plot(ts + w/2 + 0.3, hs_mse, 'o--', color=c_hs, markersize=4, linewidth=1.2)
    ax_mse.errorbar(ts - w/2 - 0.3, h_mse, yerr=h_mse_std, fmt='none',
                    ecolor='#555', capsize=2, linewidth=0.7)
    ax_mse.set_ylabel('MSE ($\\times 10^{-2}$)', fontsize=10)
    ax_mse.set_title('(a) MSE', fontsize=10, fontweight='bold', loc='left')
    ax_mse.set_ylim(0, 82)
    ax_mse.spines['top'].set_visible(False)
    ax_mse.spines['right'].set_visible(False)
    ax_mse.grid(axis='y', linewidth=0.3, alpha=0.2)
    ax_mse.tick_params(axis='both', labelsize=9)

    # === (b) R² ===
    ax_r2.bar(ts - w/2 - 0.3, h_r2, width=w, color=c_h, alpha=0.8,
              edgecolor='white', linewidth=0.5)
    ax_r2.bar(ts + w/2 + 0.3, hs_r2, width=w, color=c_hs, alpha=0.8,
              edgecolor='white', linewidth=0.5)
    ax_r2.plot(ts - w/2 - 0.3, h_r2, 'o--', color=c_h, markersize=4, linewidth=1.2)
    ax_r2.plot(ts + w/2 + 0.3, hs_r2, 'o--', color=c_hs, markersize=4, linewidth=1.2)
    ax_r2.set_ylabel('$R^2$', fontsize=10)
    ax_r2.set_xlabel('Sequence Length $T$', fontsize=10)
    ax_r2.set_title('(b) $R^2$', fontsize=10, fontweight='bold', loc='left')
    ax_r2.set_xticks(ts)
    ax_r2.set_ylim(0.25, 0.85)
    ax_r2.spines['top'].set_visible(False)
    ax_r2.spines['right'].set_visible(False)
    ax_r2.grid(axis='y', linewidth=0.3, alpha=0.2)
    ax_r2.tick_params(axis='both', labelsize=9)

    # 图例放在两图中间
    handles = [
        plt.Rectangle((0,0),1,1, color=c_h, alpha=0.8),
        plt.Rectangle((0,0),1,1, color=c_hs, alpha=0.8),
    ]
    fig.legend(handles, ['Humanoid', 'HumanoidStandup'],
               loc='center', ncol=2, fontsize=9, frameon=True,
               edgecolor='gray', fancybox=True,
               bbox_to_anchor=(0.5, 0.52))

    plt.subplots_adjust(hspace=0.25, top=0.93, bottom=0.10, left=0.12, right=0.96)
    plt.savefig('paper/figures/seqlen_sensitivity.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('paper/figures/seqlen_sensitivity.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print('Done: seqlen_sensitivity.pdf')

if __name__ == '__main__':
    gen_ablation()
    gen_seqlen()
