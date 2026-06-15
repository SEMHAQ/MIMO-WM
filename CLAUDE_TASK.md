你是一个学术论文改进专家。你的任务是迭代改进 paper/main.tex 这篇CTA期刊论文，直到ARS审稿分数超过85分。

当前状态：论文11页，9表6图，D4RL数据。ARS审稿得分52.2/100。

按优先级做：
1. 运行 python3 scripts/multiseed_train.py 做多种子实验(3 seeds)，将结果更新到论文，所有指标报告mean±std
2. 去重复：关键结果重复了4+遍，精简
3. 推理时间统一到T=32
4. 尝试MPC递推模式提升频率
5. 加GRU-WM基线到对比表

规则：不删图表只改数据和文字，数据必须真实，每次改完编译验证(xelatex main.tex)，完成后跑ARS审稿。

先读 paper/main.tex 和 paper/peer_review/ 下的审稿报告。
