---
title: SSM-World-Model
---

# 面向人形机器人状态预测的轻量级状态空间世界模型

**周新民 &nbsp; 余焕杰** — 湖南工商大学 / 湘江实验室

*控制理论与应用 (Control Theory & Applications)*, 2026 专刊: 具身智能与人形机器人

[:fontawesome-brands-github: GitHub](https://github.com/SEMHAQ/SSM-World-Model){ .md-button .md-button--primary }
[:fontawesome-solid-book: 论文PDF](https://github.com/SEMHAQ/SSM-World-Model/blob/main/paper/main.pdf){ .md-button }

---

## 研究背景

人形机器人要控制自己的身体，需要一个**世界模型**来预测："如果我施加这个动作，身体状态会怎么变化？" 然后用 **MPC（模型预测控制）** 反复问世界模型，搜索最优动作序列。

问题在于：**世界模型的推理速度决定了MPC能不能实时运行。**

| 世界模型方法 | 推理时间 | 能否用于实时MPC？ |
|:---:|:---:|:---:|
| LSTM | 27.8ms | :material-close: 太慢 |
| Transformer | >100ms | :material-close: 更慢 |
| Mamba | 4.5ms | :material-check: 可以，但部署难 |
| **SSM-WM（本文）** | **3.8ms** | :material-check: **又快又好部署** |

---

## 核心思路

!!! info "一句话概括"
    用标准 PyTorch 实现的轻量级 SSM 世界模型，推理比 LSTM 快 7 倍，精度在 MuJoCo 上反超 LSTM 6%，接上 MPC 实现 5.1Hz 实时控制。

---

## 核心结果

=== "MuJoCo 精度"

    | 方法 | MSE (×10⁻³) | R² |
    |:---:|:---:|:---:|
    | Mamba-WM | 0.821 | 0.598 |
    | **SSM-WM** | **0.834** | **0.592** |
    | LSTM-WM | 0.889 | 0.566 |
    | Transformer-WM | 0.956 | 0.528 |

    SSM-WM 比 LSTM 好 **6%**，比 Transformer 好 **13%**，与 Mamba 差距 **<2%**。

=== "推理速度"

    | 方法 | 时间 (ms) | 加速比 |
    |:---:|:---:|:---:|
    | **SSM-WM** | **3.8** | **基准** |
    | Mamba-WM | 4.5 | 0.8× |
    | LSTM-WM | 27.8 | 7.3× |
    | Transformer-WM | >100 | >26× |

=== "MPC 控制"

    | 数据集 | SSM-WM-MPC | LSTM-MPC |
    |:---:|:---:|:---:|
    | 合成数据集 | **5.1Hz** | 0.7Hz |
    | MuJoCo Humanoid | **2.1Hz** | 0.35Hz |

---

## 为什么不用 Mamba？

两者数学基础相同，区别在于实现方式：

| | SSM-WM（本文） | Mamba |
|---|:---:|:---:|
| 实现依赖 | 标准 PyTorch | 自定义 CUDA 算子 |
| 部署难度 | :material-check-circle: 低 | :material-close-circle: 高 |
| 推理速度 | **3.8ms** | 4.5ms |
| 参数量 | **0.24M** | 0.28M |
| MuJoCo MSE | 0.834 | 0.821 |

:material-star: 为了 1.5% 的精度差距（统计不显著），不值得增加部署复杂度。

---

## 文档导航

| 章节 | 内容 |
|---|---|
| [背景概念](method/background.md) | 什么是世界模型？什么是SSM？LSTM/Transformer/Mamba对比 |
| [SSM-WM架构](method/architecture.md) | 模型结构、门控机制、训练方式 |
| [MPC集成](method/mpc.md) | 世界模型如何接入模型预测控制 |
| [实验设置](experiments/setup.md) | 数据集、基线方法、评价指标 |
| [主要结果](experiments/results.md) | 合成数据集和MuJoCo的对比实验 |
| [消融实验](experiments/ablation.md) | 门控、残差、网络深度、损失函数的贡献 |
| [讨论](discussion/discussion.md) | 精度-速度权衡、与Mamba的比较 |
| [局限性](discussion/limitations.md) | 未来工作和开源信息 |
