---
title: 训练目标
---

# 训练目标

模型架构确定后，MIMO-WM 采用 **单步预测损失** 与 **多步展开损失** 相结合的训练策略，使模型既能保证单步精度，又对多步自回归展开时的累积误差具有鲁棒性——这对 MPC 规划中的多步前向模拟尤为重要。

## 复合损失函数

MIMO-WM 的总损失由两部分组成：

$$
\left\{
\begin{aligned}
\mathcal{L}_{\mathrm{s}} &= \frac{1}{T}\sum_{t=0}^{T-1}\|\mathbf{s}_{t+1} - \hat{\mathbf{s}}_{t+1}\|^2, \\[4pt]
\mathcal{L}_{\mathrm{m}} &= \frac{1}{H}\sum_{h=0}^{H-1}\|\mathbf{s}_{T+h} - \hat{\mathbf{s}}_{T+h}\|^2, \\[4pt]
\mathcal{L} &= \mathcal{L}_{\mathrm{s}} + \lambda\, \mathcal{L}_{\mathrm{m}}.
\end{aligned}
\right.
$$

### 单步预测损失 $\mathcal{L}_{\mathrm{s}}$

直接比较模型在教师强制（teacher forcing）下的单步预测值与真实值，约束模型的 **单步预测精度**。

$$
\mathcal{L}_{\mathrm{s}} = \frac{1}{T}\sum_{t=0}^{T-1}\|\mathbf{s}_{t+1} - \hat{\mathbf{s}}_{t+1}\|^2.
$$

### 多步展开损失 $\mathcal{L}_{\mathrm{m}}$

将模型自身的预测结果作为下一步输入，自回归地继续向后展开 $H$ 步，与真实轨迹比较，鼓励模型在多步展开时 **保持稳定性**。

$$
\mathcal{L}_{\mathrm{m}} = \frac{1}{H}\sum_{h=0}^{H-1}\|\mathbf{s}_{T+h} - \hat{\mathbf{s}}_{T+h}\|^2.
$$

!!! tip "为什么要多步损失？"
    多步损失在展开过程中逐步引入模型自身的预测误差，使模型学会"自己的误差会怎样累积"，从而：

    - 抑制误差爆炸（error blow-up）；
    - 让长程预测轨迹更贴近真实；
    - 直接服务于 MPC 的多步前向模拟需求。

### 权重系数 $\lambda$

总损失为

$$
\mathcal{L} = \mathcal{L}_{\mathrm{s}} + \lambda\, \mathcal{L}_{\mathrm{m}}.
$$

其中 $\lambda$ 控制多步损失在总损失中的权重。实验中通过验证集网格搜索确定 $\lambda = 0.5$、$H = 8$，在单步精度与多步稳定性之间取得最佳平衡。

| $\lambda$（多步权重） | 单步 MSE | 多步 MSE（$H=8$） | 说明 |
|---|---|---|---|
| 0（仅单步损失） | 较低 | 较高 | 单步准但多步漂移 |
| **0.5（默认）** | **平衡** | **平衡** | **最佳折中** |
| 1.0 | 略升 | 最低 | 过度强调长程 |

## 优化器与调度

### AdamW 优化器

MIMO-WM 采用 AdamW 优化器（解耦权重衰减）：

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 学习率 | $5 \times 10^{-4}$ | 主实验配置 |
| 权重衰减 | $1 \times 10^{-4}$ | 解耦的 L2 正则 |
| 优化器 | AdamW | 动量自适应 + 解耦权重衰减 |
| 批大小 | 1024 | 大批量提升训练稳定性与吞吐 |

### 余弦退火调度

学习率采用 **余弦退火（cosine annealing）** 调度，从初始值 $5 \times 10^{-4}$ 平滑衰减到接近 0：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T_{\max}}\right).
$$

余弦退火避免了阶梯式调度在边界处的震荡，有助于训练后期的精细收敛。

### 梯度裁剪

为防止训练初期梯度爆炸，对梯度范数进行裁剪，阈值为 1.0：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 完整训练配置

!!! summary "训练超参数汇总"

    | 参数 | 值 |
    |------|-----|
    | 优化器 | AdamW |
    | 学习率 | $5 \times 10^{-4}$ |
    | 权重衰减 | $1 \times 10^{-4}$ |
    | 学习率调度 | 余弦退火 |
    | 批大小 | 1024 |
    | 训练轮数 | 100 epoch |
    | 序列长度 | $T = 32$ |
    | 梯度裁剪阈值 | 1.0 |
    | 单步损失权重 | 1 |
    | 多步损失权重 $\lambda$ | 0.5 |
    | 多步展开步数 $H$ | 8 |
    | 随机种子 | 42, 123, 456, 789, 1024（5 个） |

## 训练流程伪代码

```python
# MIMO-WM 训练伪代码
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for s_batch, a_batch in dataloader:           # s, a: [B, T, d_s], [B, T, d_a]
        # --- 单步损失（教师强制）---
        s_pred_single = model.forward_single(s_batch, a_batch)
        loss_s = mse(s_pred_single, s_batch[:, 1:])

        # --- 多步损失（自回归展开 H 步）---
        s_pred_multi = model.unroll(s_batch, a_batch, H=8)
        loss_m = mse(s_pred_multi, s_truth_multi)

        # --- 复合损失 ---
        loss = loss_s + 0.5 * loss_m

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

!!! note "数据归一化"
    输入数据经 z-score 归一化（按训练集统计量），使各维度量纲一致，加速收敛并保证数值稳定。验证集使用训练集的均值和方差进行归一化，避免数据泄漏。

## 训练硬件与可复现性

- **硬件**：NVIDIA RTX 3090 GPU
- **种子**：每组实验使用 5 个随机种子（42, 123, 456, 789, 1024），报告均值与标准差
- **早停**：以验证集 MSE 最优 epoch 的模型作为最终结果（`best_epoch`）

---

上一节：[理论分析](theory.md) · 下一节：[MPC 规划](mpc.md)
