---
title: MIMO-WM 架构
---

# MIMO-WM 架构

MIMO-WM 的核心是 **多输入多输出状态空间模型（Multi-Input Multi-Output SSM, MIMO-SSM）** 架构。传统 SSM 为单输入单输出（SISO）结构，每个隐藏维度独立维护一个标量状态；MIMO-SSM 将其扩展为多输入多输出结构，并引入 sigmoid 门控机制增强表达能力。本节详述整体架构、编码器/解码器、MIMO 块定义与前向推理流程。

## 整体结构

MIMO-WM 采用"编码器—MIMO 主干—解码器"的三段式结构：

```
输入：历史状态 + 动作序列  [s_{0:T-1}; a_{0:T-1}]
              ↓
   ┌──────────────────────────┐
   │  编码器 Encoder           │  W1 → GELU → W2，映射到 D 维隐空间
   ├──────────────────────────┤
   │  MIMO 主干 × L 层          │  LayerNorm → DiagSSM → 门控 → 残差
   ├──────────────────────────┤
   │  解码器 Decoder            │  残差预测 ŝ_T = s_{T-1} + W_d·z_{T-1}
   └──────────────────────────┘
              ↓
输出：预测的下一时刻状态  ŝ_T
```

MIMO-SSM 为输入序列 $\mathbf{z}_t \in \mathbb{R}^D$ 的 **每个输入维度** $d=1,\ldots,D$ 分配一个独立的 $N$ 维隐状态向量 $\mathbf{h}_t^{(d)} \in \mathbb{R}^N$，通过 $D$ 个并行 SSM 同时处理全部维度。每个 SSM 具有独立参数 $(\mathbf{A}^{(d)}, \mathbf{B}^{(d)}, \mathbf{C}^{(d)})$，允许多样的动力学特性。

!!! abstract "核心思想"
    MIMO-SSM 通过 $D$ 个并行对角 SSM 同时建模 $D$ 个输入通道，再以 sigmoid 门控根据输入内容动态调节信息流动，在保持线性计算复杂度的同时实现更强的表达能力。

## 编码器

编码器将当前状态和动作拼接后投影到 $D$ 维隐空间，采用两层线性变换加 GELU 激活：

$$
\left\{
\begin{aligned}
\mathbf{z}_t' &= \mathbf{W}_1[\mathbf{s}_t;\, \mathbf{a}_t] + \mathbf{b}_1, \\
\mathbf{z}_t &= \mathbf{W}_2\,\mathrm{GELU}(\mathbf{z}_t') + \mathbf{b}_2.
\end{aligned}
\right.
$$

其中：

- $[\mathbf{s}_t;\, \mathbf{a}_t] \in \mathbb{R}^{d_s + d_a}$ 为拼接的状态与动作，
- $\mathbf{W}_1 \in \mathbb{R}^{D \times (d_s + d_a)}$、$\mathbf{W}_2 \in \mathbb{R}^{D \times D}$ 为权重矩阵，
- $\mathrm{GELU}(\cdot) = x\,\Phi(x)$ 为高斯误差线性单元，
- $D$ 为隐空间维度（默认 $D=96$）。

GELU 相比 ReLU 更平滑，有助于训练稳定。

## MIMO 块

MIMO 块是 MIMO-WM 的核心组件，每个块依次执行 **层归一化 → 并行对角 SSM → 门控融合 → 残差连接** 四步。

!!! definition "MIMO 块定义"
    给定输入 $\mathbf{z}_t \in \mathbb{R}^D$，MIMO 块依次执行：

    **第 1 步：层归一化**

    $$
    \tilde{\mathbf{z}}_t = \mathrm{LayerNorm}(\mathbf{z}_t).
    $$

    **第 2 步：并行对角 SSM** —— 归一化后 $\tilde{\mathbf{z}}_t$ 的每个维度 $d$ 由一个独立 SSM 处理：

    $$
    \mathbf{h}_t^{(d)} = \bar{\mathbf{A}}^{(d)}\mathbf{h}_{t-1}^{(d)} + \bar{\mathbf{B}}^{(d)}\tilde{z}_t^{(d)},
    $$

    其中 $\bar{\mathbf{A}}^{(d)} \in \mathbb{C}^{N \times N}$ 为第 $d$ 维的对角状态矩阵，各维度参数独立。

    **第 3 步：sigmoid 门控** —— SSM 输出经门控调节：

    $$
    \mathbf{o}_t = \mathbf{h}_t \odot \sigma(\mathbf{W}_g\tilde{\mathbf{z}}_t + \mathbf{b}_g),
    $$

    其中 $\mathbf{W}_g \in \mathbb{R}^{D \times D}$ 为门控权重，$\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘积。

    **第 4 步：残差连接**

    $$
    \mathbf{z}_t' = \mathbf{z}_t + \mathbf{W}_o\mathbf{o}_t + \mathbf{b}_o.
    $$

### 门控机制的作用

没有门控时，SSM 对所有信息一视同仁；引入门控后，模型能根据输入内容动态控制每个维度的信息保留比例：

- $\sigma \to \mathbf{0}$：SSM 输出被完全抑制（丢弃噪声/无关信息）；
- $\sigma \to \mathbf{1}$：退化为无门控 SSM（完全保留 SSM 预测）。

消融实验证明：移除门控后 MSE 增加 **8.9%**（详见[消融实验](../experiments/ablation.md)）。门控在 MuJoCo 接触动力学场景下贡献尤为显著——因为接触力存在更多不连续，需要门控选择性过滤信息。

```python
# MIMO 块伪代码
def mimo_block(z):
    z_norm = LayerNorm(z)                          # 第 1 步：层归一化
    h = DiagSSM(z_norm, A_bar, B_bar, C_bar)       # 第 2 步：D 路并行对角 SSM
    gate = sigmoid(W_g @ z_norm + b_g)             # 第 3 步：sigmoid 门控
    o = h * gate                                   #   逐元素融合
    z_out = z + W_o @ o + b_o                      # 第 4 步：残差连接
    return z_out
```

## 解码器

解码器将最后一层 MIMO 块的输出映射回状态空间，并采用 **残差预测**（预测状态增量而非绝对状态）：

$$
\hat{\mathbf{s}}_T = \mathbf{W}_d\mathbf{z}_{T-1}' + \mathbf{b}_d + \mathbf{s}_{T-1}.
$$

!!! tip "为什么用残差预测？"
    将预测目标建模为状态增量 $\hat{\mathbf{s}}_T - \mathbf{s}_{T-1}$，使网络只需学习相对变化，数值幅度更小、更易收敛，同时也更符合物理直觉（速度、加速度本质上是状态增量）。

## 前向推理流程

MIMO-WM 的完整前向推理过程如下：

=== "步骤 1：编码"

    $$
    \mathbf{z} \leftarrow \mathrm{Encoder}([\mathbf{s}_{0:T-1};\, \mathbf{a}_{0:T-1}]).
    $$

    将长度为 $T$ 的状态-动作序列编码为隐空间表示 $\mathbf{z} \in \mathbb{R}^{T \times D}$。

=== "步骤 2：MIMO 主干（$\ell = 1, \ldots, L$ 层）"

    对每一层 $\ell$：

    - 归一化：$\tilde{\mathbf{z}} \leftarrow \mathrm{LayerNorm}(\mathbf{z})$
    - 并行 SSM：$\mathbf{h} \leftarrow \mathrm{DiagSSM}(\tilde{\mathbf{z}})$
    - 门控：$\mathbf{o} \leftarrow \mathbf{h} \odot \sigma(\mathbf{W}_g\tilde{\mathbf{z}} + \mathbf{b}_g)$
    - 残差：$\mathbf{z} \leftarrow \mathbf{z} + \mathbf{W}_o\mathbf{o} + \mathbf{b}_o$

=== "步骤 3：解码"

    $$
    \hat{\mathbf{s}}_T \leftarrow \mathbf{s}_{T-1} + \mathrm{Decoder}(\mathbf{z}_{T-1}).
    $$

    取序列最后一步的隐表示，通过残差预测得到下一时刻状态。

## 双模式计算

MIMO-SSM 的 $D$ 个并行 SSM 支持两种等价的计算模式（等价性证明见[理论分析](theory.md)）：

| 模式 | 单层复杂度 | 实现方式 | 适用场景 |
|------|-----------|---------|---------|
| **卷积（FFT）** | $O(T\log T)$ | 批处理 FFT 一次性处理整条序列 | 批量训练、长序列 |
| **递推** | $O(TN)$ | 逐步更新隐状态 | 在线推理、低延迟 |

$D$ 个通道的卷积核 $K^{(d)}$ 可组合为张量，通过 **批处理 FFT** 一次性完成全部计算，这是 MIMO-WM 训练高效的关键。

!!! note "部署时的优势"
    部署时切换为递推模式，单步推理复杂度为 $O(N)$（与序列长度无关），延迟恒定，适合实时控制。训练时切换为卷积模式，整条序列并行计算，吞吐高。

## 默认配置

| 超参数 | 符号 | 默认值 |
|--------|------|--------|
| 隐空间维度 | $D$ | 96 |
| SSM 状态维度 | $N$（`d_state`） | 16 |
| MIMO 层数 | $L$ | 2 |
| 序列长度 | $T$ | 32 |
| 总参数量 | — | **0.138 M** |

该配置在 Humanoid 数据集上取得最优精度 MSE $= 19.87 \times 10^{-2}$，且参数量在所有对比方法中最少。

---

上一节：[背景概念](background.md) · 下一节：[理论分析](theory.md)
