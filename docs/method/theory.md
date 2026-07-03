---
title: 理论分析
---

# 理论分析

本节系统分析 MIMO-SSM 的理论性质。由于 MIMO-SSM 为每个输入维度 $d=1,\ldots,D$ 分配一个独立的 $N$ 维对角 SSM，其参数为 $(\mathbf{A}^{(d)}, \mathbf{B}^{(d)}, \mathbf{C}^{(d)})$，其中 $\mathbf{A}^{(d)} = \mathrm{diag}(a_1^{(d)}, \ldots, a_N^{(d)}) \in \mathbb{C}^{N \times N}$，因此 **稳定性、离散化与计算模式均可归约为单通道对角 SSM 的分析**，再在 $D$ 个通道上并行重复。

## 单通道对角 SSM

设单通道对角 SSM 的隐状态 $\mathbf{h}(t) \in \mathbb{C}^N$，输入 $x(t) \in \mathbb{R}$，输出 $y(t) \in \mathbb{R}$，其连续时间形式为

$$
\left\{
\begin{aligned}
\dot{\mathbf{h}}(t) &= \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), \\
y(t) &= \mathbf{C}\mathbf{h}(t) + \mathbf{D}x(t).
\end{aligned}
\right.
$$

其中 $\mathbf{A} = \mathrm{diag}(a_1, \ldots, a_N)$，$a_n = -\alpha_n + j\beta_n$，$\alpha_n > 0$ 确保渐近稳定。经零阶保持离散化后：

$$
\left\{
\begin{aligned}
\mathbf{h}_k &= \bar{\mathbf{A}}\mathbf{h}_{k-1} + \bar{\mathbf{B}}x_k, \\
y_k &= \mathbf{C}\mathbf{h}_k + \mathbf{D}x_k.
\end{aligned}
\right.
$$

其中 $\bar{\mathbf{A}} = \mathrm{diag}(\bar{a}_1, \ldots, \bar{a}_N)$，$\bar{a}_n = \exp(\Delta a_n)$，$|\bar{a}_n| < 1$ 保证离散系统稳定。MIMO-SSM 将上述结构在 $D$ 个通道上并行复制，总参数量为 $D$ 倍单通道参数量。

### 稳定性

!!! theorem "渐近稳定性"
    对角参数化 $a_n = -\alpha_n + j\beta_n$（$\alpha_n > 0$）保证了系统的渐近稳定性。

    **证明。** 离散化后 $\bar{a}_n = \exp(\Delta a_n) = \exp(-\Delta\alpha_n)\exp(j\Delta\beta_n)$，其模长

    $$
    |\bar{a}_n| = \exp(-\Delta\alpha_n) < 1,
    $$

    即所有特征值严格位于单位圆内，离散系统渐近稳定。$\square$

    工程含义：$\alpha_n$ 通过可学习参数 $\mathrm{softplus}^{-1}$ 或 $\log$ 参数化保证为正，无需额外约束即可训练时保持稳定。

## 两种计算模式

!!! definition "定义：递推模式与卷积模式"
    单通道对角 SSM 有两种等价计算模式，MIMO-SSM 在 $D$ 个通道上并行执行。

    **递推模式**：每个通道 $d$ 逐步计算

    $$
    \left\{
    \begin{aligned}
    \mathbf{h}_t^{(d)} &= \bar{\mathbf{A}}^{(d)}\mathbf{h}_{t-1}^{(d)} + \bar{\mathbf{B}}^{(d)}x_t^{(d)}, \\
    y_t^{(d)} &= \mathbf{C}^{(d)}\mathbf{h}_t^{(d)} + \mathbf{D}x_t^{(d)}.
    \end{aligned}
    \right.
    $$

    **卷积模式**：通过卷积核

    $$
    K^{(d)}[t] = \sum_{n} C_n^{(d)}\bar{b}_n^{(d)}(\bar{a}_n^{(d)})^t
    $$

    并行计算

    $$
    y_t^{(d)} = \sum_{\tau=0}^{t} K^{(d)}[t-\tau]\, x_\tau^{(d)} + \mathbf{D}x_t^{(d)}.
    $$

    $D$ 个通道的 $K^{(d)}$ 可组合为张量，通过 **批处理 FFT** 一次性完成全部计算。

## 定理一：双模式等价性

!!! theorem "定理 1（递推与卷积等价性）"
    对每个通道 $d$，递推模式与卷积模式产生 **完全相同的输出**。MIMO-SSM 的 $D$ 个通道并行运行，输出为各通道结果的拼接。

    **证明。** 对任意通道 $d$，由递推式展开隐状态：

    $$
    \mathbf{h}_t^{(d)} = \sum_{\tau=0}^{t} (\bar{\mathbf{A}}^{(d)})^{t-\tau}\bar{\mathbf{B}}^{(d)}x_\tau^{(d)}.
    $$

    代入输出方程得

    $$
    \begin{aligned}
    y_t^{(d)} &= \mathbf{C}^{(d)}\mathbf{h}_t^{(d)} + \mathbf{D}x_t^{(d)} \\
    &= \sum_{\tau=0}^{t} \underbrace{\mathbf{C}^{(d)}(\bar{\mathbf{A}}^{(d)})^{t-\tau}\bar{\mathbf{B}}^{(d)}}_{K^{(d)}[t-\tau]}\, x_\tau^{(d)} + \mathbf{D}x_t^{(d)},
    \end{aligned}
    $$

    与卷积模式一致。$D$ 个通道相互独立，同时成立。$\square$

!!! note "工程意义"
    等价性意味着：**训练时** 使用卷积模式（FFT 并行，吞吐高），**部署时** 切换为递推模式（单步 $O(1)$ 延迟，恒定），两者输出严格一致，无需任何微调。这是 MIMO-WM 兼顾训练效率与部署延迟的理论基础。

## 定理二：计算复杂度

!!! theorem "定理 2（计算复杂度）"
    设序列长度 $T$，SSM 状态维度 $N$，隐空间维度 $D$，层数 $L$。对角 SSM 单层的递推模式和卷积模式的时间复杂度分别为 $O(TN)$ 和 $O(T\log T)$。MIMO-WM 整体复杂度为

    $$
    O\!\left(LTD\log T + LTD^2 + TD(d_s+d_a)\right).
    $$

    作为对比，LSTM 和 Transformer 的复杂度分别为 $O(TLD^2)$ 和 $O(T^2LD + TLD^2)$。当序列较长（$T > D/\log D$）时，MIMO-WM 的计算效率最优。

    **证明。**

    **(i) 单层 SSM。** 递推模式每步执行 $N$ 维向量运算，共 $T$ 步，故为 $O(TN)$；卷积模式通过 FFT 计算 $T$ 点循环卷积，复杂度 $O(T\log T)$。当 $N > \log_2 T$ 时卷积更优——本文 $N=16 > \log_2 32 = 5$。

    **(ii) MIMO-WM 各模块。**

    | 模块 | 复杂度 |
    |------|--------|
    | 编码器 | $O(TD(d_s+d_a))$ |
    | 单个 MIMO 块 | $O(TD\log T + TD^2)$ |
    | $L$ 层 MIMO 块 | $O(LTD\log T + LTD^2)$ |
    | 解码器 | $O(TDd_s)$ |

    求和即得整体复杂度 $O(LTD\log T + LTD^2 + TD(d_s+d_a))$。$\square$

!!! example "与基线复杂度对比"
    | 方法 | 序列建模复杂度 | 本文 $T{=}32, D{=}96$ 时 |
    |------|---------------|-------------------------|
    | LSTM | $O(TLD^2)$ | $32 \times 2 \times 96^2 \approx 5.9 \times 10^5$ |
    | Transformer | $O(T^2LD + TLD^2)$ | $32^2 \times 2 \times 96 + \cdots \approx 2.0 \times 10^6$ |
    | **MIMO-WM** | $O(TD\log T + TD^2)$ | $32 \times 96 \times 5 + 32 \times 96^2 \approx 3.1 \times 10^5$ |

    MIMO-WM 的 $O(TD\log T)$ 远优于 Transformer 的 $O(T^2D)$，本文 $T=32$、$D=96$ 时约快 6 倍。

## 定理三：CEM-MPC 收敛性

将 MIMO-WM 嵌入模型预测控制后，采用交叉熵方法（Cross-Entropy Method, CEM）求解动作序列。定义动作序列 $\mathbf{a} \in \mathbb{R}^{Hd_a}$ 的代价函数：

$$
J(\mathbf{a}) = \sum_{h=1}^{H}\Big[\|\hat{\mathbf{s}}_{h} - \mathbf{s}_{\mathrm{ref}}\|_{\mathbf{Q}}^2 + \|\mathbf{a}_{h-1}\|_{\mathbf{R}}^2\Big],
$$

其中 $\hat{\mathbf{s}}_h$ 为 MIMO-WM 的第 $h$ 步预测，$\mathbf{s}_{\mathrm{ref}}$ 为参考状态，$\mathbf{Q}$、$\mathbf{R}$ 为权重矩阵。MPC 在每个时刻求解 $\min_{\|\mathbf{a}\|_\infty \leqslant a_{\max}} J(\mathbf{a})$。

!!! assumption "假设 1（代价函数的正则性）"
    代价函数 $J(\mathbf{a})$ 在有界闭集 $\mathcal{A} \subset \mathbb{R}^{Hd_a}$ 上连续可微，且存在 Lipschitz 常数 $L_J > 0$ 使得 $\|\nabla J(\mathbf{a})\| \leqslant L_J$ 对任意 $\mathbf{a} \in \mathcal{A}$ 成立。进一步假设 $J$ 在全局最优解 $\mathbf{a}^*$ 附近满足 $\mu$-强凸性，即

    $$
    J(\mathbf{a}) \geqslant J(\mathbf{a}^*) + \frac{\mu}{2}\|\mathbf{a} - \mathbf{a}^*\|^2.
    $$

!!! theorem "定理 3（CEM-MPC 收敛性）"
    在假设 1 下，CEM-MPC 算法具有以下收敛性质：

    **(a) 单调性。** 精英样本的期望代价单调非增：

    $$
    \mathbb{E}[J(\mathbf{a}_{\mathrm{best}}^{(k)})] \leqslant \mathbb{E}[J(\mathbf{a}_{\mathrm{best}}^{(k-1)})].
    $$

    **(b) 指数收敛。** 期望代价以指数速率收敛到全局最优：

    $$
    \mathbb{E}[J(\mathbf{a}_{\mathrm{best}}^{(k)})] - J^* \leqslant \frac{\mu}{2}\|\boldsymbol{\Sigma}^{(0)}\|_F^2 \exp\!\left(-\frac{N_e}{K}k\right).
    $$

    **(c) 分布收敛。** 采样分布 $\mathcal{N}(\boldsymbol{\mu}^{(k)}, \boldsymbol{\Sigma}^{(k)})$ 弱收敛到退化分布 $\delta_{\mathbf{a}^*}$。

    **证明。**

    设第 $k$ 轮采样集 $\mathcal{S}^{(k)}$ 含 $K$ 个独立同分布样本，其全样本方差估计为

    $$
    \boldsymbol{\Sigma}_{\mathcal{S}}^{(k)} = \frac{1}{K-1}\sum_{\mathbf{a} \in \mathcal{S}^{(k)}}(\mathbf{a} - \bar{\boldsymbol{\mu}})(\mathbf{a} - \bar{\boldsymbol{\mu}})^{\mathsf{T}}.
    $$

    精英集 $E^{(k)} \subset \mathcal{S}^{(k)}$ 包含代价最小的 $N_e$ 个样本，其方差 $\boldsymbol{\Sigma}^{(k+1)}$ 是全样本方差的子集估计，Frobenius 范数以概率 $1 - N_e/K$ 缩减。

    **(a) 单调性。** 由方差递减，采样分布逐步集中于代价更低的次水平集内，故期望代价单调非增。

    **(b) 指数收敛。** 由 $\mu$-强凸性，$J(\mathbf{a}) \geqslant J^* + \frac{\mu}{2}\|\mathbf{a} - \mathbf{a}^*\|^2$。设 $\|\boldsymbol{\Sigma}^{(0)}\|_F^2 = \sigma_0^2$，由方差递推

    $$
    \begin{aligned}
    \|\boldsymbol{\Sigma}^{(k)}\|_F^2 &\leqslant \sigma_0^2\left(1 - \frac{N_e}{K}\right)^{2k} \\
    &\leqslant \sigma_0^2 \exp\!\left(-2\frac{N_e}{K}k\right).
    \end{aligned}
    $$

    代入强凸不等式取期望即得指数收敛速率 $\exp(-N_e k / K)$。

    **(c) 分布收敛。** 由 $\boldsymbol{\Sigma}^{(k)} \to \mathbf{0}$，$\boldsymbol{\mu}^{(k)} \to \mathbf{a}^*$，故 $\mathcal{N}(\boldsymbol{\mu}^{(k)}, \boldsymbol{\Sigma}^{(k)})$ 弱收敛到退化分布 $\delta_{\mathbf{a}^*}$。$\square$

!!! info "收敛速率的工程解读"
    收敛速率 $\exp(-N_e k / K)$ 表明：精英比例 $N_e/K$ 越大（但 $N_e$ 过大会保留劣质样本），收敛越快。本文取 $K=256$、$N_e=32$（精英比例 12.5%）、$M=5$ 轮迭代，在收敛速度与样本利用率之间取得平衡（详见 [MPC 实验](../experiments/mpc.md)）。

## 推论：门控机制的可学习性

!!! corollary "推论（门控梯度）"
    门控输出为 $\mathbf{o}_t = \mathbf{h}_t \odot \sigma(\mathbf{W}_g\tilde{\mathbf{z}}_t + \mathbf{b}_g)$。当 $\sigma \to \mathbf{0}$ 时 SSM 输出被完全抑制，当 $\sigma \to \mathbf{1}$ 时退化为无门控 SSM。门控参数的梯度为

    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{W}_g} = \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} \odot \mathbf{h}_t \odot \sigma'(\cdot)\, \tilde{\mathbf{z}}_t^{\mathsf{T}},
    $$

    其中 $\sigma'$ 为 sigmoid 的导数。该梯度受 SSM 输出 $\mathbf{h}_t$ 调控，使门控能 **自适应学习何时信任 SSM 预测**。

## 参数量分析

综合以上分析，MIMO-WM 的总参数量为

$$
N_{\mathrm{total}} = O\!\left(LD^2 + D(d_s + d_a)\right).
$$

| 模块 | 参数量 |
|------|--------|
| 编码器 $\mathbf{W}_1, \mathbf{W}_2$ | $D(d_s + d_a) + D^2$ |
| 单层 MIMO 块（SSM + 门控 $\mathbf{W}_g$ + 输出 $\mathbf{W}_o$） | $O(DN + D^2)$ |
| $L$ 层 MIMO 块 | $O(LD^2)$ |
| 解码器 $\mathbf{W}_d$ | $Dd_s$ |

默认配置 $D=96$、$N=16$、$L=2$ 时总参数量约 **0.138 M**，在所有对比方法（LSTM 0.227M、GRU 0.190M、Transformer 0.302M、Mamba 0.224M、TCN 0.189M）中最为轻量。

---

上一节：[MIMO-WM 架构](architecture.md) · 下一节：[训练目标](training.md)
