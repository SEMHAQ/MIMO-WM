---
title: MPC 规划
---

# 基于 MIMO-WM 的模型预测控制

将训练好的 MIMO-WM 嵌入模型预测控制（Model Predictive Control, MPC）框架，验证其在规划决策中的可行性。本节介绍 MPC 的代价函数、梯度 MPC 与 CEM 采样式 MPC 两种求解方法，以及 CEM 的收敛性保证。

## 什么是模型预测控制？

模型预测控制是一种基于模型的优化控制方法：在每个控制时刻，通过世界模型 **前向模拟** 一段有限时域的未来轨迹，搜索使代价最小的动作序列，**只执行第一个动作**，然后滚动到下一时刻重新规划。

```
当前状态 s_T
    ↓
世界模型前向展开：给定动作序列 [a_T, ..., a_{T+H-1}]
    预测未来 H 步轨迹 ŝ_{T+1}, ..., ŝ_{T+H}
    ↓
优化器：调整动作序列，最小化代价 J(a)
    ↓
执行：只执行第一个动作 a_T*
    ↓
滚动到下一时刻，重复
```

!!! info "MPC 的核心依赖"
    MPC 需要 **大量** 前向预测。每次规划都要反复问世界模型"如果这样做会怎样？"，因此世界模型的推理速度直接决定 MPC 的控制频率。

## 代价函数

在每个控制时刻 $T$，求解以下有限时域优化问题：

$$
\min_{\mathbf{a}_{T:T+H-1}}\ J(\mathbf{a}) = \sum_{h=1}^{H}\Big[\underbrace{\|\hat{\mathbf{s}}_{T+h} - \mathbf{s}_{\mathrm{ref}}\|_{\mathbf{Q}}^2}_{\text{状态跟踪误差}} + \underbrace{\|\mathbf{a}_{T+h-1}\|_{\mathbf{R}}^2}_{\text{控制能量}}\Big].
$$

其中：

- $\hat{\mathbf{s}}_{T+h}$ 为 MIMO-WM 预测的第 $h$ 步未来状态，
- $\mathbf{s}_{\mathrm{ref}}$ 为目标参考状态（如期望的站立姿态），
- $\mathbf{Q}$、$\mathbf{R}$ 为权重矩阵，权衡跟踪精度与控制能量，
- $H$ 为预测时域（horizon，默认 $H=10$）。

本文实现两种 MPC 求解方法。

## 方法一：梯度 MPC

使用 Adam 优化器对动作序列进行 $I$ 次梯度下降迭代。每次迭代通过 MIMO-WM **前向展开** 计算代价 $J(\mathbf{a})$，再 **反向传播** 更新动作序列。

```python
# 梯度 MPC 伪代码
a = initialize_action_sequence(H, d_a)         # 可学习动作序列
a.requires_grad_(True)
opt = Adam([a], lr=action_lr)

for i in range(I):                              # I 次迭代（约 30~50）
    s_pred = mimo_wm.unroll(s_T, a, H)          # 前向展开 H 步
    J = quad_cost(s_pred, s_ref, a, Q, R)       # 计算代价
    J.backward()                                # 反向传播到 a
    opt.step()
    a.data.clamp_(-1, 1)                        # 动作约束

a_T_star = a[0]                                 # 只执行第一步
```

!!! warning "梯度 MPC 的开销"
    梯度 MPC 需要反向传播，在 SSM 类模型中计算开销较大。预实验表明 $I \approx 50$ 在跟踪精度与计算时间之间取得较好平衡，但仍较慢（详见 [MPC 实验](../experiments/mpc.md)）。

## 方法二：CEM 采样式 MPC

梯度 MPC 的反向传播开销大，而 **交叉熵方法（Cross-Entropy Method, CEM）** 仅需前向传播评估候选动作序列，天然适合 GPU 并行计算。

### 算法步骤

=== "第 1 步：初始化"

    初始化采样分布的均值与方差：

    $$
    \boldsymbol{\mu} \leftarrow \mathbf{0}_{H \times d_a}, \quad
    \boldsymbol{\sigma} \leftarrow \mathbf{1}_{H \times d_a}, \quad
    \epsilon \leftarrow 10^{-6}.
    $$

=== "第 2 步：迭代优化（$m = 1, 2, \ldots, M$）"

    1. 从高斯分布采样 $K$ 个候选动作序列，并截断到 $[-1, 1]$：

       $$
       \{\mathbf{a}^{(k)}\}_{k=1}^{K} \sim \mathcal{N}(\boldsymbol{\mu},\, \mathrm{diag}(\boldsymbol{\sigma}^2)).
       $$

    2. 对每个候选序列，用 MIMO-WM 前向展开 $H$ 步，计算代价 $J^{(k)}$。

    3. 选取代价最小的 $N_e$ 个 **精英样本** 构成精英集 $E$。

    4. 用精英样本的均值与方差更新分布参数：

       $$
       \left\{
       \begin{aligned}
       \boldsymbol{\mu} &\leftarrow \frac{1}{N_e}\sum_{\mathbf{a} \in E}\mathbf{a}, \\
       \boldsymbol{\sigma} &\leftarrow \sqrt{\frac{1}{N_e}\sum_{\mathbf{a} \in E}(\mathbf{a} - \boldsymbol{\mu})^2} + \epsilon.
       \end{aligned}
       \right.
       $$

=== "第 3 步：输出"

    $$
    \mathbf{a}_T^* \leftarrow \boldsymbol{\mu}[0],
    $$

    将分布均值的第一个动作作为当前控制时刻的最优动作。

### 默认参数

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| 采样数 | $K$ | 256 | 每轮采样的候选动作序列数 |
| 精英数 | $N_e$ | 32 | 每轮保留的代价最小样本数 |
| 迭代轮数 | $M$ | 5 | CEM 迭代轮数 |
| 预测时域 | $H$ | 10 | 前向展开步数 |
| 截断阈值 | — | $[-1, 1]$ | 动作范围约束 |

!!! tip "为什么 CEM 比 梯度 MPC 快？"
    $K=256$ 个候选序列的评估通过 **GPU 批量并行** 实现，全部仅需前向传播。相比之下，梯度 MPC 的 $I$ 次迭代每次都要做一次完整的反向传播，串行且昂贵。实验表明 CEM-MPC 可将控制频率提升 1～2 个数量级。

## CEM-MPC 收敛性保证

CEM-MPC 的收敛性由 [理论分析中的定理 3](theory.md#定理三cem-mpc-收敛性) 保证：

- **(a) 单调性**：精英样本的期望代价单调非增；
- **(b) 指数收敛**：期望代价以 $\exp(-N_e k / K)$ 的速率收敛到全局最优；
- **(c) 分布收敛**：采样分布弱收敛到集中于最优解 $\mathbf{a}^*$ 的退化分布 $\delta_{\mathbf{a}^*}$。

收敛速率 $\exp(-N_e k / K)$ 中，精英比例 $N_e/K = 32/256 = 12.5\%$，配合 $M=5$ 轮迭代，能在收敛速度与样本利用率之间取得平衡。

## MPC 实验结果预览

| 方法 | 梯度 MPC / Hz | CEM-MPC / Hz | 3 步预测 MSE |
|------|--------------|--------------|-------------|
| LSTM | 1.45 | 21.16 | 0.262 |
| Mamba | 0.60 | 9.79 | 0.231 |
| **MIMO-WM** | 0.51 | 7.99 | **0.216** |

CEM-MPC 相比梯度 MPC 可将控制频率提升 1～2 个数量级；MIMO-WM 的短期预测精度优于 Mamba-WM 与 LSTM-WM，能有效转化为 MPC 规划质量。详见 [MPC 实验](../experiments/mpc.md)。

---

上一节：[训练目标](training.md) · 下一节：[实验设置](../experiments/setup.md)
