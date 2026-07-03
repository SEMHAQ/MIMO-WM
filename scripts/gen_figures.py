"""生成论文图3(消融实验)和图4(序列长度敏感性)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

# 统一中文字体
zhfont = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=10)
zhfont_s = FontProperties(fname='/mnt/c/Windows/Fonts/simhei.ttf', size=9)

# 统一配色方案
C_BLUE = '#2E86AB'
C_RED = '#E74C3C'
C_PURPLE = '#A23B72'
C_ZONE = '#27ae60'

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
        ('w/o 门控',      20.36, 0.10, 0.745, 0.142),
        ('w/o 残差',      26.14, 0.26, 0.673, 0.208),
        ('w/o 层归一化',  21.25, 0.48, 0.734, 0.208),
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

    BAR = C_BLUE
    HL = C_RED
    colors = [HL if i == 0 else BAR for i in range(n)]

    fig = plt.figure(figsize=(7.0, 8.5))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.18, wspace=0.12,
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
    ax_mse.text(0.5, 1.02, '(a) MSE', transform=ax_mse.transAxes,
                fontproperties=zhfont, fontsize=11, fontweight='bold', ha='center', va='bottom')
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
    ax_r2.text(0.5, 1.02, '(b) $R^2$', transform=ax_r2.transAxes,
                fontproperties=zhfont, fontsize=11, fontweight='bold', ha='center', va='bottom')
    ax_r2.spines['top'].set_visible(False)
    ax_r2.spines['right'].set_visible(False)
    ax_r2.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_r2.tick_params(axis='x', labelsize=9)
    plt.setp(ax_r2.get_yticklabels(), visible=False)

    # (c) 参数量
    bubble_sizes = [p * 500 for p in params]
    ax_params.scatter(params, y, s=bubble_sizes, c=colors, edgecolors='white',
                      linewidth=0.8, zorder=3, alpha=0.85)
    for i in range(n):
        fw = 'bold' if i == 0 else 'normal'
        ax_params.text(params[i] + 0.015, y[i], f'{params[i]:.3f}',
                       fontsize=8, va='center', color='#222', fontweight=fw)
    ax_params.set_xlabel('参数量 (M)', fontproperties=zhfont, fontsize=11)
    ax_params.set_xlim(0, 0.75)
    ax_params.text(0.5, 1.02, '(c) 参数量', transform=ax_params.transAxes,
                fontproperties=zhfont, fontsize=11, fontweight='bold', ha='center', va='bottom')
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_visible(False)
    ax_params.grid(axis='x', linewidth=0.3, alpha=0.2)
    ax_params.tick_params(axis='x', labelsize=9)

    # Y轴
    ax_mse.invert_yaxis()
    ax_params.set_ylim(ax_mse.get_ylim())

    ax_mse.set_yticks(y)
    ax_mse.set_yticklabels(labels, fontsize=9)
    for i, tick in enumerate(ax_mse.get_yticklabels()):
        tick.set_fontproperties(zhfont)
        if i == 0:
            tick.set_fontweight('bold')
            tick.set_color(HL)

    ax_params.set_yticks(y)
    ax_params.set_yticklabels(labels, fontsize=9)
    for i, tick in enumerate(ax_params.get_yticklabels()):
        tick.set_fontproperties(zhfont)
        if i == 0:
            tick.set_fontweight('bold')
            tick.set_color(HL)

    plt.savefig('paper/figures/ablation_results.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('paper/figures/ablation_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print('Done: ablation_results.pdf')

# ============================================================
def gen_seqlen():
    x = np.arange(5)
    w = 0.35
    ts_labels = ['8', '16', '32', '64', '128']

    h_mse = [20.14, 19.23, 21.18, 21.28, 41.13]
    h_r2 = [0.765, 0.764, 0.735, 0.708, 0.448]

    hs_mse = [50.02, 51.29, 54.65, 58.80, 64.00]
    hs_r2 = [0.478, 0.461, 0.428, 0.380, 0.320]

    c_h = C_BLUE
    c_hs = C_PURPLE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6.5))
    fig.patch.set_facecolor('white')

    # ============ (a) MSE ============
    ax1.bar(x - w/2, h_mse, w, color=c_h, alpha=0.5, edgecolor='none', zorder=2)
    ax1.bar(x + w/2, hs_mse, w, color=c_hs, alpha=0.5, edgecolor='none', zorder=2)
    ax1.plot(x - w/2, h_mse, 'o-', color=c_h, linewidth=1.5, markersize=5, zorder=3)
    ax1.plot(x + w/2, hs_mse, 's-', color=c_hs, linewidth=1.5, markersize=5, zorder=3)
    # 统一推荐区间
    ax1.axvspan(-0.5, 1.5, alpha=0.10, color=C_ZONE, zorder=1)

    ax1.set_ylabel('MSE', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ts_labels)
    ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.5, 1.02, '(a) 预测MSE随序列长度的变化', transform=ax1.transAxes,
             fontproperties=zhfont, fontsize=10, ha='center', va='bottom')

    # ============ (b) R2 ============
    ax2.bar(x - w/2, h_r2, w, color=c_h, alpha=0.5, edgecolor='none', zorder=2)
    ax2.bar(x + w/2, hs_r2, w, color=c_hs, alpha=0.5, edgecolor='none', zorder=2)
    ax2.plot(x - w/2, h_r2, 'o-', color=c_h, linewidth=1.5, markersize=5, zorder=3)
    ax2.plot(x + w/2, hs_r2, 's-', color=c_hs, linewidth=1.5, markersize=5, zorder=3)
    ax2.axvspan(-0.5, 1.5, alpha=0.10, color=C_ZONE, zorder=1)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax2.set_xlabel('序列长度 T', fontproperties=zhfont, fontsize=10)
    ax2.set_ylabel('$R^2$', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ts_labels)
    ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.5, 1.02, '(b) $R^2$随序列长度的变化', transform=ax2.transAxes,
             fontproperties=zhfont, fontsize=10, ha='center', va='bottom')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, top=0.90, bottom=0.10)

    # 3-item legend: 贴近上图底部
    legend_elements = [
        Line2D([0], [0], color=c_h, marker='o', linewidth=1.5, markersize=5, label='Humanoid'),
        Line2D([0], [0], color=c_hs, marker='s', linewidth=1.5, markersize=5, label='HumanoidStandup'),
        mpatches.Patch(facecolor=C_ZONE, alpha=0.15, label='推荐区间'),
    ]
    fig.legend(handles=legend_elements, loc='center', ncol=3, fontsize=9, prop=zhfont_s,
               bbox_to_anchor=(0.5, 0.5), frameon=True, fancybox=True,
               framealpha=0.9, edgecolor='gray')
    plt.savefig('paper/figures/seqlen_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/seqlen_sensitivity.png', dpi=300, bbox_inches='tight')
    print('Done: seqlen_sensitivity.pdf')

if __name__ == '__main__':
    gen_ablation()
    gen_seqlen()
