# MIMO-WM 补充材料 / Supplementary Material

本目录是论文《MIMO-WM: 面向人形机器人的轻量级状态空间世界模型》（稿件编号 CCTA260363）
修改稿所涉补充实验的原始数据、评测脚本与部署产物，供审阅与复现使用。

本目录中每一个结果文件都由本目录 `scripts/` 下的脚本直接生成，未做手工后处理；
`results/` 中保存的是生成该结果当时的原始输出。

---

## 1. 目录结构

```
revision_experiments/
├── README.md              本说明
├── results/               实验原始输出
│   ├── matrix_results.json        对比矩阵（5 seeds × 2 数据集）
│   ├── gpu_time_scaling.json      GPU 推理时间—序列长度伸缩
│   ├── train_peak_mem.json        训练峰值显存
│   ├── trunc_control.json         长序列截断对照
│   ├── trunc_longbudget.json      长序列 + 训练预算对照
│   ├── cem_evolution.json         CEM 代价与采样方差演化
│   ├── mpc_extended.json          MPC 控制频率与 3 步预测精度
│   ├── cpu_deploy_bench.json      x86 CPU 部署基准
│   ├── deploy_kernel_verify.json  部署核等价性与 ONNX 导出验证
│   ├── resource_ledger.json       FLOPs 与权重体积账本
│   ├── mimo_wm_deploy.onnx        导出用于部署评测的 ONNX 模型
│   └── ckpt/                      各数据集各随机种子的最优权重
└── scripts/               评测脚本
```

---

## 2. 运行环境

| 用途 | 环境 |
|---|---|
| GPU 训练 / 推理 | Python 3.10, PyTorch 2.1.2+cu121, NVIDIA RTX 3090 |
| CPU / ONNX 部署基准 | Python 3.x, PyTorch (CPU), onnx 1.21, onnxruntime 1.23 |
| 机载实测 | 少林派 SLKY01 开发板, 算能 BM1684（八核 aarch64）, ONNXRuntime 1.16.3, 纯 CPU 推理 |

除机载平台实测（第 5.7 节，结果已汇总于修改稿表 6）外，其余实验均在 RTX 3090 上完成。

---

## 3. 数据准备

实验数据由 Gymnasium 的 MuJoCo 环境接口采集，仓库中不直接分发（体积原因），
可用仓库根目录的脚本重新生成：

```bash
python3 scripts/generate_data.py
```

该脚本对 `Humanoid-v5` 与 `HumanoidStandup-v5` 各采集 1000 条 episode，
按 8:2 划分训练集与验证集，并写入 `data/humanoid/` 与 `data/humanoid_standup/`。
输入按 z-score 归一化，归一化统计量仅由训练集计算。

---

## 4. 结果文件与修改稿的对应关系

| `results/` 文件 | 修改稿位置 | 生成脚本 |
|---|---|---|
| `matrix_results.json` | 表 1、表 2（Humanoid / HumanoidStandup 主对比） | `run_revision_matrix.py` |
| `gpu_time_scaling.json` | 第 5.2 节末段（吞吐口径计算伸缩性） | `bench_gpu_scaling.py` |
| `train_peak_mem.json` | 第 5.7 节（训练峰值显存对照） | `mem_train_peak.py` |
| `trunc_control.json` | 第 5.4 节（截断对照） | `trunc_control.py` |
| `trunc_longbudget.json` | 第 5.4 节（$T{=}128$ 增加训练预算对照） | `trunc_control.py --epochs 250` |
| `cem_evolution.json` | 第 5.5 节（CEM 代价与采样方差演化） | `cem_evolution.py` |
| `mpc_extended.json` | 表 5（MPC 控制性能对比） | `mpc_extended.py` |
| `cpu_deploy_bench.json` | 第 5.7 节（x86 CPU 与机载平台延迟对照） | `bench_deploy_cpu.py` |
| `deploy_kernel_verify.json` | 第 5.7 节（部署核等价性与 ONNX 对拍） | `deploy_mimo.py` |
| `resource_ledger.json` | 第 5.7 节（单窗 FLOPs 与权重体积账本） | `resource_ledger.py` |
| `mimo_wm_deploy.onnx` | 第 5.7 节（导出模型，opset 17） | `deploy_mimo.py` |
| `ckpt/*.pt` | 表 1、表 2 中 MIMO-WM 各行（2 数据集 × 5 种子的最优验证权重） | `run_revision_matrix.py` |

修改稿表 3（消融实验）与表 4（序列长度敏感性）来自仓库中既有的实验脚本，
不在本补充材料范围内。

---

## 5. 复现方法

所有脚本均以仓库根目录为工作目录运行（脚本内按 `revision_experiments/results/...`
相对路径写出结果）。GPU 脚本需要可用的 CUDA 设备。

```bash
# 对比矩阵：表 1 / 表 2（2 数据集 × 6 个模型 × 5 seeds，可断点续跑）
python3 revision_experiments/scripts/run_revision_matrix.py

# 指定子集运行
python3 revision_experiments/scripts/run_revision_matrix.py \
    --jobs MIMO-WM MIMO-WM-noGate S4D-WM LRU-WM Performer-WM Transformer-Reg \
    --datasets humanoid humanoid_standup --seeds 42 123 456 789 1024

# MPC 控制性能：表 5（依赖 ckpt/ 中已有的 MIMO-WM 权重）
python3 revision_experiments/scripts/mpc_extended.py

# GPU 时间伸缩 / 训练峰值显存 / CEM 演化
python3 revision_experiments/scripts/bench_gpu_scaling.py
python3 revision_experiments/scripts/mem_train_peak.py
python3 revision_experiments/scripts/cem_evolution.py

# 长序列截断对照（默认 100 epoch；加 --epochs 250 得 trunc_longbudget）
python3 revision_experiments/scripts/trunc_control.py --seeds 42 123
python3 revision_experiments/scripts/trunc_control.py --seeds 42 123 --epochs 250

# 部署核等价性验证 + ONNX 导出
python3 revision_experiments/scripts/deploy_mimo.py

# x86 CPU 部署基准；资源账本
python3 revision_experiments/scripts/bench_deploy_cpu.py
python3 revision_experiments/scripts/resource_ledger.py
```

---

## 6. 结果口径说明

以下几点影响对数字的解读，特此说明：

1. **统一配置。** 表 1、表 2、表 5 中所有模型均为隐空间维度 $D{=}96$、层数 $L{=}2$；
   MIMO-WM 另取状态维度 $N{=}16$。每个配置以 5 个随机种子（42, 123, 456, 789, 1024）
   运行，报告均值与标准差。训练为 100 epoch、AdamW、学习率 $5\times10^{-4}$、
   余弦退火、批量大小 1024、梯度裁剪阈值 1.0。
   表中标 `d` 的常规规模 Transformer 为独立配置（$D{=}192$、多头、$L{=}3$、FFN$=4D$）。

2. **推理时间的测量口径。** 表 1、表 2 的"时间"列为 $B{=}1$、$T{=}32$、
   在空闲 GPU 上测得的单次前向延迟。该口径下调度与固定开销占主导，
   不同模型之间的差异不反映渐进复杂度；因此计算伸缩性的结论以**批量吞吐口径**
   （$B{=}128$，`gpu_time_scaling.json`）给出，两者不可混用。

3. **部署延迟的测量口径。** 机载平台的延迟（表 6）为 ONNXRuntime 纯 CPU 推理的
   单窗延迟中位数，测试期间温度恒定 44.5 ℃、未见降频。同一模型在 x86 服务器上的
   延迟约为机载平台的 1/8 至 1/9。若直接使用 PyTorch 的卷积或复数递推路径在 CPU 上
   测时，会得到与渐进复杂度相反的结论（见 `cpu_deploy_bench.json`），故部署数字
   统一取 ONNXRuntime 口径，并在修改稿中标注了测试环境。

4. **MPC 频率的口径。** 表 5 的控制频率由 GPU 上并行评估 256 条候选序列测得。
   在算力受限的机载平台上按同一 CEM 配置单次规划耗时约 5.82 s（约 0.17 Hz）；
   将候选规模缩减至 32、迭代 3 轮、时域 5 后降至 0.42 s（约 2.41 Hz）。
   该对比说明规划频率与候选规模近似成正比，机载端需要相应缩减候选规模。

5. **部署核与训练形态的一致性。** `deploy_mimo.py` 实现的纯实数单步递推部署核由训练
   所用卷积模式反推得到，与训练卷积路径的输出最大误差约 $2.4\times10^{-7}$
   （$T$ 为 8 与 32、多个随机初始化），ONNX 导出后 ONNXRuntime 与 PyTorch 部署核
   误差约 $1.2\times10^{-7}$。即部署形态与训练形态在浮点精度内一致，无需微调。

---

## 7. 权重文件

`results/ckpt/` 保存 MIMO-WM 在 2 个数据集、5 个随机种子下的最优验证权重
（共 10 个文件，每个约 0.55 MB，对应 0.138M 参数）。这些权重供表 1、表 2 的
MIMO-WM 各行以及 `cem_evolution.py`、`trunc_control.py` 复用。

---

## 8. 许可

补充材料随论文一并提供，供审阅与复现使用。
