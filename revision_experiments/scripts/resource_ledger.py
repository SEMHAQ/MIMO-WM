"""资源开销: 参数量/权重体积/单窗 FLOPs(profiler 实测) + Jetson Orin Nano 级延迟预估(标注为预估).

用法: python revision_experiments/scripts/resource_ledger.py   (CPU 即可, 不占 GPU)
产物: revision_experiments/results/resource_ledger.json
"""
import os, sys, json, torch
sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel
from src.models.baselines import TransformerWorldModel, LSTMWorldModel, TCNWorldModel
from src.models.ssm_world_model import SSMWorldModel

SD, AD, T = 348, 17, 32


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def flops_forward(m, sd, ad, T, reps=5):
    st = torch.randn(1, T, sd)
    ac = torch.randn(1, T - 1, ad)
    with torch.no_grad():
        for _ in range(reps):
            m(st, ac)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            m(st, ac)
    total = 0
    for e in prof.key_averages():
        f = getattr(e, 'flops', 0) or 0
        total += f
    return total


def main():
    torch.set_num_threads(4)
    models = {
        'MIMO-WM': MIMOWorldModel(SD, AD, d_model=96, d_state=16, n_layers=2),
        'S4D-WM(no-gate SSM-WM)': SSMWorldModel(SD, AD, d_model=96, d_state=16, n_layers=2),
        'Transformer-WM': TransformerWorldModel(SD, AD, d_model=96, nhead=4, n_layers=2),
        'Transformer-Reg(D192L3)': TransformerWorldModel(SD, AD, d_model=192, nhead=6, n_layers=3),
        'LSTM-WM': LSTMWorldModel(SD, AD, hidden_dim=96, n_layers=2),
        'TCN-WM': TCNWorldModel(SD, AD, d_model=96, n_layers=2),
    }
    out = {'config': f'state_dim={SD}, action_dim={AD}, T={T}', 'models': {}}
    for name, m in models.items():
        m.eval()
        p = nparams(m)
        fl = flops_forward(m, SD, AD, T)
        per_step = fl / T
        out['models'][name] = {
            'params_m': round(p / 1e6, 4),
            'weight_bytes_mb': round(p * 4 / 1e6, 3),
            'window_flops_T32': int(fl),
            'per_step_flops': int(per_step),
        }
        print(f'{name:<26} params={p/1e6:.3f}M  weight={p*4/1e6:.2f}MB  window_FLOPs={fl/1e6:.2f}M  per-step={per_step/1e3:.1f}K', flush=True)

    # Jetson Orin Nano 级延迟: 不做纯 FLOP 投影(微模型为延迟/开销主导, FLOP 投影会给出
    # 无意义的 µs 级数字). 板级延迟以机载实测为准(见表 6 与机载实测说明.md); 本文件仅提供
    # FLOPs/权重/激活规模的资源开销, x86 侧延迟见 results/cpu_onnx_bench.json.
    out['note'] = ('FLOP-only projection omitted: tiny models are latency/overhead-bound, '
                   'projected ~us numbers are meaningless. On-board latency requires real measurement; '
                   'x86 reference: see results/cpu_onnx_bench.json '
                   '(ONNXRuntime CPU, T=8 median 0.34 ms, 1 thread).')
    os.makedirs('revision_experiments/results', exist_ok=True)
    with open('revision_experiments/results/resource_ledger.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print('Saved -> revision_experiments/results/resource_ledger.json (无 FLOP 纯投影, 见 note)', flush=True)


if __name__ == '__main__':
    main()
