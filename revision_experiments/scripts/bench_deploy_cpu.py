"""CPU 端部署基准 (A5): 测量各世界模型在纯 CPU 上的推理时延与内存占用.

用途:"嵌入式/机载硬件实测：推理耗时、显存占用".

设计说明 (与正文口径对齐):
- 统一配置与论文主对比一致: D=96, L=2, N=16; Humanoid 口径 state_dim=348, action_dim=17.
- 原子成本单位 = "一次完整模型前向 (B=1, 给定 T 窗口)", 这正是 MPC 单次 rollout 一步/单次决策评估的成本.
- conv  = 训练/批量推理默认的 FFT 卷积模式 (DiagSSM mode='conv');
  recurrent = 论文承诺的部署递推模式 (mode='recurrent', O(1) 单步), 仅 SSM/MIMO 类支持.
- 线程数 1 / 4 模拟低功耗/嵌入式核数; 记录峰值 RSS 增量 (psutil, 缺失则跳过).
- 本脚本只需架构(随机权重)即可测量时延/内存, 不依赖 GPU/checkpoint.

用法:  python revision_experiments/scripts/bench_deploy_cpu.py
产物:  revision_experiments/results/cpu_deploy_bench.json
"""
import os, sys, json, time, platform
import torch

sys.path.insert(0, os.path.abspath('.'))

STATE_DIM, ACTION_DIM = 348, 17          # Humanoid
DM, DS, NL = 96, 16, 2                    # 论文主配置 D=96 N=16 L=2
THREADS = [1, 4]
TS = [8, 32]

from src.models.mimo_world_model import MIMOWorldModel
from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel, TCNWorldModel
from src.models.mamba_world_model import MambaWorldModel


def make_models():
    m = {}
    m['MIMO-WM'] = MIMOWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, d_state=DS, n_layers=NL)
    # MIMO-WM 去掉 sigmoid 门控 = S4D 风格 "LayerNorm + DiagSSM + 残差", 即 "MIMO-SSM(无门控)"
    m['MIMO-WM(no-gate)'] = SSMWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, d_state=DS, n_layers=NL)
    m['Transformer-WM'] = TransformerWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, nhead=4, n_layers=NL)
    m['LSTM-WM'] = LSTMWorldModel(STATE_DIM, ACTION_DIM, hidden_dim=DM, n_layers=NL)
    m['GRU-WM'] = GRUWorldModel(STATE_DIM, ACTION_DIM, hidden_dim=DM, n_layers=NL)
    m['TCN-WM'] = TCNWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, n_layers=NL)
    m['Mamba-WM'] = MambaWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, n_layers=NL)
    return m


def latency(fn, reps_warm=3, reps=50):
    for _ in range(reps_warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def peak_rss_delta(fn):
    try:
        import psutil
    except Exception:
        return None
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    fn()
    after = proc.memory_info().rss
    return max(0, (after - before)) / 1e6  # MB


def main():
    torch.set_num_threads(1)  # 先占位, 每轮会重设
    models = make_models()
    out = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'state_dim': STATE_DIM, 'action_dim': ACTION_DIM,
        'config': {'d_model': DM, 'd_state': DS, 'n_layers': NL},
        'results': {},
    }
    for name, model in models.items():
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        rec = {'params_m': round(params, 3)}
        print(f'\n=== {name} (params={params:.3f}M) ===', flush=True)
        for nthreads in THREADS:
            torch.set_num_threads(nthreads)
            for T in TS:
                states = torch.randn(1, T, STATE_DIM)
                actions = torch.randn(1, T - 1, ACTION_DIM)
                tag = f'conv_ms_t{T}_th{nthreads}'
                try:
                    rec[tag] = round(latency(lambda: model(states, actions)), 3)
                    print(f'  conv    T={T:3d} th={nthreads} : {rec[tag]:.3f} ms', flush=True)
                except Exception as e:
                    rec[tag] = None
                    print(f'  conv    T={T:3d} th={nthreads} : FAIL {e}', flush=True)
                # 递推部署模式 (仅支持 mode 参数者)
                if hasattr(model, 'forward') and 'mode' in getattr(model, 'forward', lambda *a, **k: {}).__code__.co_varnames:
                    try:
                        with torch.no_grad():
                            rec[f'recurrent_ms_t{T}_th{nthreads}'] = round(latency(
                                lambda: model(states, actions, mode='recurrent')), 3)
                    except Exception as e:
                        rec[f'recurrent_ms_t{T}_th{nthreads}'] = None
        # 峰值内存 (单次 conv 前向 RSS 增量, 线程4)
        torch.set_num_threads(4)
        states = torch.randn(1, TS[-1], STATE_DIM)
        actions = torch.randn(1, TS[-1] - 1, ACTION_DIM)
        rec['rss_delta_mb'] = peak_rss_delta(lambda: model(states, actions))
        rec['weight_bytes_mb'] = round(params * 4, 3)  # fp32 权重体积估算
        out['results'][name] = rec

    os.makedirs('revision_experiments/results', exist_ok=True)
    path = 'revision_experiments/results/cpu_deploy_bench.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {path}', flush=True)


if __name__ == '__main__':
    main()
