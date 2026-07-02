import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

old = r"""训练损失函数包含单步预测损失和多步展开损失:"""

new = r"""MIMO-WM的训练损失由单步预测损失和多步展开损失组成:"""

content = content.replace(old, new)

old2 = r"""其中$\lambda$为多步损失权重, $H$为展开步数.
单步损失确保模型在每个时间步的预测准确性,
多步损失则鼓励模型在自回归展开时保持预测稳定性.
实验中设置$\lambda=0.5$, $H=8$.

多步展开损失在训练时使用真实状态作为输入(teacher forcing),
而在MPC部署时使用模型自身的预测进行自回归展开, 存在分布不匹配.
多步损失通过展开训练使模型逐步适应自身的预测误差, 从而缓解此问题.
实验中通过网格搜索确定$\lambda{=}0.5$, $H{=}8$.
优化采用AdamW优化器\textsuperscript{\cite{27}}(学习率$1{\times}10^{-3}$, 权重衰减$1{\times}10^{-4}$),
余弦退火调度, 梯度裁剪阈值1.0."""

new2 = r"""其中$\mathcal{L}_{\text{s}}$约束单步预测精度, $\mathcal{L}_{\text{m}}$鼓励模型在多步自回归展开时保持稳定性, $\lambda$为权重系数.
多步损失在展开过程中逐步引入模型自身的预测误差, 使模型对累积误差具有鲁棒性, 这对MPC规划中的多步前向模拟尤为重要.
实验中通过验证集网格搜索确定$\lambda=0.5$, $H=8$.
优化采用AdamW优化器\textsuperscript{\cite{27}}, 学习率$1\times10^{-3}$, 权重衰减$1\times10^{-4}$, 余弦退火调度, 梯度裁剪阈值1.0."""

content = content.replace(old2, new2)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
