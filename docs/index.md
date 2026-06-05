---
title: SSM-World-Model
---

# 面向人形机器人状态预测的轻量级状态空间世界模型

**周新民 &nbsp; 余焕杰** — 湖南工商大学 / 湘江实验室

*控制理论与应用*, 2026

[:fontawesome-brands-github: GitHub](https://github.com/SEMHAQ/SSM-World-Model){ .md-button .md-button--primary }

---

## 一句话概括

> 用一种轻量级的序列模型（SSM）给机器人建一个"世界模型"，让机器人能预测自己身体的状态变化，再结合MPC做控制决策。

## 这篇文章做了什么？

```
人形机器人想控制自己的身体
    ↓
需要一个"世界模型"来预测：如果我做这个动作，身体会怎样？
    ↓
以前用LSTM建世界模型 → 太慢，来不及控制
   用Transformer建   → 更慢，O(T²)复杂度
   用Mamba建         → 快，但部署需要特殊CUDA算子
    ↓
我们用SSM（和Mamba类似的数学基础，但实现更简单）
    ↓
效果：比LSTM快7倍，比Transformer快26倍，精度不差
    ↓
接上MPC控制器 → 实现了5.1Hz的实时控制
