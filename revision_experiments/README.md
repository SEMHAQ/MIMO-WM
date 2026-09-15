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
│   ├── official_s4d.json          S4D-WM 官方参考实现对照（5 seeds × 2 数据集）
│   ├── official_lru.json          LRU-WM 官方参考实现对照（5 seeds × 2 数据集）
│   ├── nogate_seed_robustness.json 无门控变体的补种子复核（10 种子口径）
│   ├── init_variants.json         门控偏置初始化的变体对照
│   ├── ablation_crosscheck.json   消融数值的跨文件一致性核对
│   ├── s4d_fairness.json          已发表 S4D-WM 的实现差异溯源
│   ├── gpu_time_scaling.json      GPU 推理时间—序列长度伸缩
│   ├── train_peak_mem.json        训练峰值显存
│   ├── trunc_control.json         长序列截断对照
│   ├── trunc_longbudget.json      长序列 + 训练预算对照
│   ├── cem_evolution.json         CEM 代价与采样方差演化
│   ├── mpc_extended.json          MPC 控制频率与 3 步预测精度
│   ├── mpc_official.json          MPC 口径下 S4D-WM / LRU-WM 的官方实现重跑
│   ├── cpu_deploy_bench.json      x86 CPU 部署基准
│   ├── deploy_kernel_verify.json  部署核等价性与 ONNX 导出验证
│   ├── resource_ledger.json       FLOPs 与权重体积账本
│   ├── onboard_bench.json         机载首批测量（每模型一进程，MIMO-WM 四窗口共享一进程）
│   ├── onboard_bench_console.txt  机载首批测量的终端输出留档
│   ├── onboard_bench_mpc.json     机载平台 CEM-MPC 规划耗时实测
│   ├── mimo_wm_deploy.onnx        导出用于部署评测的 ONNX 模型
│   └── ckpt/                      各数据集各随机种子的最优权重
├── scripts/               评测脚本（GPU / x86 侧）
└── onboard_bench/         机载平台实测脚本、模型与结果
    ├── bench_sbc.py               机载推理延迟与内存测速
    ├── bench_mpc_sbc.py           机载 CEM-MPC 规划耗时测速
    ├── export_bench_models.py     在 x86 上导出待测 ONNX 模型与参考输入/输出
    ├── export_mpc_onnx.py         导出支持动态批量的 MIMO-WM ONNX（供 MPC 测速）
    ├── reference_io.npz           参考输入/输出，用于机载数值一致性核对
    ├── models/                    机载测速所用 ONNX 模型（15 个）与 manifest.json
    ├── results/                   机载逐窗口原始结果 JSON（表 6 的来源）
    └── 机载实测说明.md            机载实测的平台、口径、复现命令与全部结果
```

> **两处需要留意的版本关系。** 其一，`results/onboard_bench.json` 与 `onboard_bench_console.txt`
> 是机载的**首批**测量：那批每模型一个进程，但 MIMO-WM 的四个序列长度共享同一进程，
> 故其 `peak_rss_MB` 是四窗口累计峰值、不能归到模型头上。表 6 与修改稿第 5.7 节一律采用
> `onboard_bench/results/` 下**逐窗口单进程**重跑的这批结果。其二，`results/true_lru.json` 与
> `scripts/run_true_lru.py` 是早期对 LRU 的一次**已作废**尝试，它把 GLU 挂在递归之前、
> 把 B/C 写成实对角，与官方实现不符；表 1、表 2 中的 LRU-WM 取自 `official_lru.json`，
> 与这两个文件无关。二者保留于此仅为便于追溯，请勿引用。

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
| `matrix_results.json` | 表 1、表 2（Humanoid / HumanoidStandup 主对比；含 MIMO-WM、无门控、Performer-WM 与常规规模 Transformer） | `run_revision_matrix.py` |
| `official_s4d.json` | 表 1、表 2 中 S4D-WM 各行 | `run_official_s4d.py` |
| `official_lru.json` | 表 1、表 2 中 LRU-WM 各行 | `run_official_lru.py` |
| `gpu_time_scaling.json` | 第 5.2 节末段（吞吐口径计算伸缩性） | `bench_gpu_scaling.py` |
| `train_peak_mem.json` | 第 5.7 节（训练峰值显存对照） | `mem_train_peak.py` |
| `trunc_control.json` | 第 5.4 节（截断对照） | `trunc_control.py` |
| `trunc_longbudget.json` | 第 5.4 节（$T{=}128$ 增加训练预算对照） | `trunc_control.py --epochs 250` |
| `cem_evolution.json` | 第 5.5 节（CEM 代价与采样方差演化） | `cem_evolution.py` |
| `mpc_extended.json` | 表 5（MPC 控制性能对比；除 S4D-WM 与 LRU-WM 外的各行） | `mpc_extended.py` |
| `mpc_official.json` | 表 5 中 S4D-WM 与 LRU-WM 两行 | `run_mpc_official.py` |
| `cpu_deploy_bench.json` | 第 5.7 节（x86 CPU 与机载平台延迟对照） | `bench_deploy_cpu.py` |
| `deploy_kernel_verify.json` | 第 5.7 节（部署核等价性与 ONNX 对拍） | `deploy_mimo.py` |
| `resource_ledger.json` | 第 5.7 节（单窗 FLOPs 与权重体积账本） | `resource_ledger.py` |
| `onboard_bench/results/bench_result_*_T*.json` | 表 6 全部数值、第 5.7 节（机载延迟、内存、温度） | `onboard_bench/bench_sbc.py`（逐窗口单进程） |
| `onboard_bench_mpc.json` | 第 5.7 节末（机载 5.82 s / 0.42 s 规划耗时） | `onboard_bench/bench_mpc_sbc.py` |
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

# S4D-WM / LRU-WM：官方参考实现的等价移植（表 1、表 2 对应行）
python3 revision_experiments/scripts/run_official_s4d.py \
    --datasets humanoid humanoid_standup --seeds 42 123 456 789 1024
python3 revision_experiments/scripts/run_official_lru.py \
    --datasets humanoid humanoid_standup --seeds 42 123 456 789 1024
# 两个脚本同时输出同口径的控制组（原实现 S4D/LRU、w/o 门控、MIMO-WM），用于核对与本目录
# 其余结果的数值一致性；脚本内使用原子写入，中断不会损坏已有结果文件。

# 核对表 1、表 2 表注 f 的 LRU 推理延迟（官方 scan/loop 两种写法 vs 其余各行）
python3 revision_experiments/scripts/check_lru_timing.py

# MPC 口径下的官方实现重跑：表 5 的 S4D-WM 与 LRU-WM 两行
python3 revision_experiments/scripts/run_mpc_official.py
# 该脚本复用 mpc_extended.py 的训练与 MPC 协议，同时重跑两个控制组（原实现 S4D/LRU）：
# 其 3 步 MSE 与 mpc_extended.json 逐位相同（0.233037 与 0.217782），可据此确认管线口径一致。

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

### 5.1 机载平台实测（表 6）

机载延迟与 MPC 规划耗时无法在 x86 上复现，须在 aarch64 板端运行。
`onboard_bench/` 即当时发往板端的完整运行包，步骤如下：

```bash
# (1) 在 x86 上重新生成待测 ONNX 模型与参考输入/输出（可选，仓库中已附）
python3 revision_experiments/onboard_bench/export_bench_models.py
python3 revision_experiments/onboard_bench/export_mpc_onnx.py

# (2) 把整个 onboard_bench/ 目录拷到板端，在板端执行
pip3 install onnxruntime numpy
cd onboard_bench
sudo sh -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > $p/scaling_governor; done'
python3 bench_sbc.py MIMO-WM        # 产出 bench_result.json
python3 bench_mpc_sbc.py            # 产出 bench_result_mpc.json
```

板端脚本自带数值正确性核对：与 `reference_io.npz` 中 x86 导出的参考输出逐元素比对，
两条日志中报告的 `diff` 即该最大绝对误差（本机载运行中 $T{=}16$ 为 $2.4\times10^{-7}$，
其余为 0）。`results/onboard_bench*.json` 即上述两条命令产出的 `bench_result*.json`，
仅重命名以区分来源，内容未作改动；终端输出见 `onboard_bench_console.txt`。

---

## 6. 结果口径说明

以下几点影响对数字的解读，特此说明：

1. **统一配置。** 表 1、表 2、表 5 中所有模型均为隐空间维度 $D{=}96$、层数 $L{=}2$；
   MIMO-WM 另取状态维度 $N{=}16$。每个配置以 5 个随机种子（42, 123, 456, 789, 1024）
   运行，报告均值与标准差。训练为 100 epoch、AdamW、学习率 $5\times10^{-4}$、
   余弦退火、训练批量大小 256、梯度裁剪阈值 1.0（评测批量大小为 1024，仅影响前向）。
   表中标 `d` 的常规规模 Transformer 为独立配置（$D{=}192$、多头、$L{=}3$、FFN$=4D$）。

2. **推理时间的测量口径。** 表 1、表 2 的"时间"列为 $B{=}1$、$T{=}32$、
   在空闲 GPU 上测得的单次前向延迟。该口径下调度与固定开销占主导，
   不同模型之间的差异不反映渐进复杂度；因此计算伸缩性的结论以**批量吞吐口径**
   （$B{=}128$，`gpu_time_scaling.json`）给出，两者不可混用。

3. **S4D-WM 与 LRU-WM 的实现口径。** 表 1、表 2 中这两行取自官方参考实现的等价移植：
   S4D 取自 `S4D-Lin` 的官方 PyTorch 实现，LRU 取自 DeepMind 的官方实现，二者均取状态维度
   $N{=}16$，参数量分别为 0.132M 与 0.150M。两行的时间列需按实现方式解读：官方 LRU 以
   逐位循环驱动对角递推（`forward_loop`）或以 Python 级递归扫描（`forward_scan`）实现，
   均未做算子融合，在 $T{=}32$ 下由启动开销主导，故其 5.82 ms 反映的是参考实现的写法，
   而非 LRU 架构本身的计算量，改用等价的循环实现亦在 5.2 至 5.5 ms（`check_lru_timing.py`）。两行同时输出同口径的
   控制组（原实现、w/o 门控、MIMO-WM），其数值与本目录其余结果逐位一致，可据此核对口径。
   表 5 中这两行同样取自官方实现（`mpc_official.json`，同表 1、表 2 的口径），其在 MPC 口径下
   的控制频率亦受上述实现方式影响：官方 LRU 为 2.74 Hz、官方 S4D 为 7.27 Hz。表 5 的
   3 步预测 MSE 不受计时影响，可作为纯精度口径使用。

4. **部署延迟与内存的测量口径。** 机载平台的延迟与内存（表 6）为 ONNXRuntime 纯 CPU
   推理的实测值（`onboard_bench/results/bench_result_*_T*.json`），测试期间温度在
   45.0 至 45.5 ℃ 之间、未见降频。表中 1 线程列为 `th1`、4 线程列为 `th4`，同一单元
   内的单步延迟由该单元中位延迟除以 $T$ 折算得到（脚本内 `per_step_ms`）。
   内存一列必须逐窗口单进程运行才可归到该模型：`VmHWM` 在进程内只增不减，一个进程装
   多个窗口时 `peak_rss_MB` 只是累计峰值。增量为加载模型并完成首帧推理前后的进程常驻
   内存之差，其中含约 38.6 MB 的 Python 与 ONNXRuntime 运行时开销；峰值随 $T$ 增大源于
   ONNX 按步展开，与权重体积无关。同一模型在 x86 服务器上的延迟约为机载平台的 1/8。
   若直接使用 PyTorch 的卷积或复数递推路径在 CPU 上测时，会得到与渐进复杂度相反的结论
   （见 `cpu_deploy_bench.json`），故部署数字统一取 ONNXRuntime 口径，并在修改稿中
   标注了测试环境。同批测得的各基线机载数值见 `onboard_bench/机载实测说明.md` 第 5.2 节，
   **修改稿正文未据此作任何对比结论**。

5. **MPC 频率的口径。** 表 5 的控制频率由 GPU 上并行评估 256 条候选序列测得。
   在算力受限的机载平台上按同一 CEM 配置单次规划耗时约 5.82 s（约 0.17 Hz）；
   将候选规模缩减至 32、迭代 3 轮、时域 5 后降至 0.42 s（约 2.41 Hz）
   （`onboard_bench_mpc.json`，每个配置测 3 个控制时刻取中位数）。
   该对比说明规划频率与候选规模近似成正比，机载端需要相应缩减候选规模。
   机载 MPC 测速以 4 线程运行，使用的是动态批量 ONNX（`MIMO-WM_T8_batch.onnx`），
   其批量输出与单样本输出逐位一致。

6. **部署核与训练形态的一致性。** `deploy_mimo.py` 实现的纯实数单步递推部署核由训练
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
