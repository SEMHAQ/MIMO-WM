"""A8: GPU 推理时间—序列长度 T 伸缩曲线 (MIMO-WM vs Transformer, 随机权重即可).
用途:展示随 T 增长 Transformer 二次 vs MIMO 拟线性的实测差距.

需 GPU 空闲时运行:
  wsl -e bash -lc "cd /mnt/e/Project/SSM-World-Model && python3 revision_experiments/scripts/bench_gpu_scaling.py"
产物: revision_experiments/results/gpu_time_scaling.json
"""
import os, sys, json, time, torch
sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel
from src.models.baselines import TransformerWorldModel

SD, AD = 348, 17
TS = [8, 16, 32, 64, 128, 256]


def lat_gpu(fn, reps):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    dev = torch.device('cuda')
    mimo = MIMOWorldModel(SD, AD, d_model=96, d_state=16, n_layers=2).to(dev).eval()
    trans = TransformerWorldModel(SD, AD, d_model=96, nhead=4, n_layers=2).to(dev).eval()
    out = {'config': f'state_dim={SD}, action_dim={AD}', 'T': TS, 'ms': {}}
    for T in TS:
        st = torch.randn(1, T, SD, device=dev)
        ac = torch.randn(1, T - 1, AD, device=dev)
        reps = 400 if T <= 32 else (200 if T <= 128 else 100)
        with torch.no_grad():
            tm = lat_gpu(lambda: mimo(st, ac), reps)
            tt = lat_gpu(lambda: trans(st, ac), reps)
        out['ms'][str(T)] = {'MIMO-WM': round(tm, 3), 'Transformer-WM': round(tt, 3),
                             'ratio_Trans_over_MIMO': round(tt / tm, 2)}
        print(f'T={T:4d}  MIMO {tm:.3f} ms | Trans {tt:.3f} ms | ratio {tt/tm:.2f}x', flush=True)
    json.dump(out, open('revision_experiments/results/gpu_time_scaling.json', 'w'), indent=2)
    print('Saved -> revision_experiments/results/gpu_time_scaling.json', flush=True)


if __name__ == '__main__':
    main()
