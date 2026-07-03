---
title: 问题描述与预备知识
---

# 问题描述与预备知识

本节首先给出机器人世界模型的形式化定义，然后回顾状态空间模型（state space model, SSM）的连续与离散形式，最后介绍对角参数化与零阶保持（zero-order hold, ZOH）离散化，为后续 MIMO-WM 架构奠定基础。

## 世界模型问题描述

设机器人在时刻 $t$ 的状态为 $\mathbf{s}_t \in \mathbb{R}^{d_s}$，执行的动作（各关节力矩或目标位置）为 $\mathbf{a}_t \in \mathbb{R}^{d_a}$。状态向量包含机身位置、速度、朝向以及各关节的角度与角速度等信息。世界模型的目标是学习状态转移函数

$$
\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t),
$$

使得给定历史观测序列 $\{\mathbf{s}_{0:T-1}, \mathbf{a}_{0:T-1}\}$，能够预测未来 $H$ 步的状态轨迹 $\hat{\mathbf{s}}_{T:T+H-1}$。

!!! info "为什么需要世界模型？"
    机器人作为具身智能的重要载体，其高维、强非线性及欠驱动特性使得"感知—决策—控制"的闭环极具挑战。世界模型作为环境动力学的内部表征，可在模型预测控制（MPC）框架中通过前向模拟多步未来状态轨迹，为优化求解提供动力学约束。其预测精度和推理速度直接决定控制性能。

在实际应用中，状态转移函数 $f$ 通常具有 **高维**、**强非线性** 与 **耦合** 三重特性，这对世界模型的精度与计算效率同时提出了严苛要求。

| 符号 | 含义 |
|------|------|
| $\mathbf{s}_t \in \mathbb{R}^{d_s}$ | 时刻 $t$ 的机器人状态 |
| $\mathbf{a}_t \in \mathbb{R}^{d_a}$ | 时刻 $t$ 的动作 |
| $d_s$ | 状态维度（Humanoid 为 348，HumanoidStandup 为 376） |
| $d_a$ | 动作维度（均为 17） |
| $T$ | 序列长度（默认 32） |
| $H$ | 多步预测时域 |

## 序列建模方法回顾

机器人状态是随时间演化的序列数据，预测"下一刻"必须理解"前面每一步"的时序模式。常见序列建模方法各有优劣：

| 方法 | 核心机制 | 复杂度 | 主要优点 | 主要缺点 |
|------|---------|--------|---------|---------|
| LSTM | 门控循环单元逐步递推 | $O(TLD^2)$ | 非线性能力强 | 无法并行，长程依赖弱 |
| GRU | 简化门控 | $O(TLD^2)$ | 参数较少 | 仍需串行计算 |
| Transformer | 自注意力 | $O(T^2LD+TLD^2)$ | 长程建模强 | 二次复杂度，长序列昂贵 |
| TCN | 因果/膨胀卷积 | $O(TLD)$ | 可并行 | 受限于卷积核感受野 |
| Mamba | 选择性扫描 SSM | $O(TLD)$ | 输入自适应 | 需自定义 CUDA 算子 |
| **SSM（本文）** | 线性递推 + FFT 卷积 | $O(TLD\log T)$ | 并行训练、$O(1)$ 单步推理 | 线性假设的局限 |

!!! note "本文选择"
    本文采用结构更简单的对角 SSM，通过标准 PyTorch（FFT）即可实现，部署门槛低，并在精度上达到与 Mamba 接近的水平（详见[实验结果](../experiments/results.md)）。

## 连续时间状态空间模型

连续时间状态空间模型定义为：

$$
\left\{
\begin{aligned}
\dot{\mathbf{h}}(t) &= \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t), \\
\mathbf{y}(t) &= \mathbf{C}\mathbf{h}(t) + \mathbf{D}\mathbf{x}(t).
\end{aligned}
\right.
$$

其中：

- $\mathbf{h}(t) \in \mathbb{R}^{N}$ 为隐状态，
- $\mathbf{x}(t)$ 为输入，
- $\mathbf{y}(t)$ 为输出，
- $\mathbf{A} \in \mathbb{R}^{N \times N}$、$\mathbf{B} \in \mathbb{R}^{N \times 1}$、$\mathbf{C} \in \mathbb{R}^{1 \times N}$、$\mathbf{D} \in \mathbb{R}$ 为系统参数。

SSM 源于控制理论中的线性系统描述：隐状态 $\mathbf{h}(t)$ 按线性微分方程演化，输出是隐状态（与输入）的线性组合。

## 零阶保持离散化

数字计算机无法直接处理连续时间方程，需对其进行离散化。采用 **零阶保持（ZOH）** 离散化后，SSM 可表示为：

$$
\left\{
\begin{aligned}
\mathbf{h}_k &= \bar{\mathbf{A}}\mathbf{h}_{k-1} + \bar{\mathbf{B}}\mathbf{x}_k, \\
\mathbf{y}_k &= \mathbf{C}\mathbf{h}_k + \mathbf{D}\mathbf{x}_k.
\end{aligned}
\right.
$$

其中离散化系数为：

$$
\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \qquad
\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}\big(\exp(\Delta \mathbf{A}) - \mathbf{I}\big) \cdot \Delta \mathbf{B},
$$

$\Delta$ 为采样步长。

!!! tip "两种计算模式"
    SSM 支持两种等价的计算模式：

    - **递推模式**：复杂度 $O(TN)$，适合在线推理，单步延迟 $O(1)$；
    - **卷积模式**：通过 FFT 在 $O(T\log T)$ 内计算，适合批量训练与长序列。

    这两种模式的等价性是 MIMO-WM 的核心理论保证之一，详见[理论分析](theory.md)。

## 对角 SSM 参数化

将状态矩阵 $\mathbf{A}$ 约束为对角形式：

$$
\mathbf{A} = \mathrm{diag}(a_1, a_2, \ldots, a_N),
$$

其中对角元素 $a_n = -\alpha_n + j\beta_n$ 为复数（$\alpha_n > 0$ 确保系统渐近稳定，$j$ 为虚数单位）。在此参数化下，SSM 的离散化系数可 **解析计算**，无需矩阵指数与求逆：

$$
\left\{
\begin{aligned}
\bar{a}_n &= \exp(\Delta \cdot a_n), \\
\bar{b}_n &= \frac{\exp(\Delta \cdot a_n) - 1}{a_n}.
\end{aligned}
\right.
$$

### 全局卷积核

对角 SSM 的输出可通过一个全局卷积核高效计算：

$$
K[t] = \sum_{n=1}^{N} C_n \cdot \bar{b}_n \cdot \bar{a}_n^{\,t},
$$

其中 $K \in \mathbb{R}^{T}$ 为长度 $T$ 的卷积核。该卷积可通过 **快速傅里叶变换（FFT）** 在 $O(T\log T)$ 时间内完成。

!!! example "为什么对角化？"
    - **稳定性可控**：$\alpha_n > 0$ 直接保证 $|\bar{a}_n| = e^{-\alpha_n \Delta} < 1$，离散系统稳定；
    - **解析离散化**：避免昂贵的矩阵指数与矩阵求逆；
    - **FFT 加速**：输出可写成线性卷积，支持 $O(T\log T)$ 并行训练。

### 与 Mamba 的关系

S4D（对角 SSM）采用固定对角矩阵，无法根据输入内容调整动力学；Mamba 引入选择性扫描实现输入自适应，但需自定义 CUDA 算子，部署门槛较高。MIMO-WM 在保留对角 SSM 计算效率的基础上，通过 **sigmoid 门控机制** 实现输入自适应的信息控制，仅需标准 PyTorch 算子即可实现（详见[架构](architecture.md)）。

| 对比维度 | S4D / 对角 SSM | Mamba | **MIMO-SSM（本文）** |
|---------|---------------|-------|---------------------|
| 参数化 | 固定对角 | 选择性扫描 | 对角 + 门控 |
| 输入自适应 | 否 | 是 | **是（门控）** |
| 实现依赖 | 标准 PyTorch | 自定义 CUDA | **标准 PyTorch** |
| 部署门槛 | 低 | 高 | **低** |

---

下一步：[MIMO-WM 架构 →](architecture.md)
