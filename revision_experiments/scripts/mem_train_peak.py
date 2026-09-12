"""A7: 训练期峰值显存实测 (前向+反向). MIMO-WM(conv) vs Transformer, 各 (B,T) 组合.
用途: 度量显存随 T 的增长 —— SSM 无 O(B·h·T²) 注意力分数矩阵, 显存近似线性.

需 GPU 空闲时运行:
  wsl -e bash -lc "cd /mnt/e/Project/SSM-World-Model && python3 revision_experiments/scripts/mem_train_peak.py"
产物: revision_experiments/results/train_peak_mem.json
"""
import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel
from src.models.baselines import TransformerWorldModel

SD, AD = 348, 17
COMBO = [(32, 256), (32, 1024), (64, 256), (128, 128)]


def peak_mb(fwd_bwd):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fwd_bwd()
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


def main():
    dev = torch.device('cuda')
    out = {'config': f'state_dim={SD}, action_dim={AD}', 'peak_mb': {}}
    for name, Model in [('MIMO-WM', MIMOWorldModel), ('Transformer-WM', TransformerWorldModel)]:
        kw = {'state_dim': SD, 'action_dim': AD, 'd_model': 96}
        kw['d_state' if 'd_state' in Model.__init__.__code__.co_varnames else 'nhead'] = 16 if 'd_state' in Model.__init__.__code__.co_varnames else 4
        kw['n_layers'] = 2
        m = Model(**kw).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        print(f'\n{name}', flush=True)
        for T, B in COMBO:
            st = torch.randn(B, T, SD, device=dev)
            ac = torch.randn(B, T - 1, AD, device=dev)
            y = torch.randn(B, SD, device=dev)
            fn = lambda: (lambda l: (l.backward(), opt.step(), opt.zero_grad()))(nn.functional.mse_loss(m(st, ac), y))
            try:
                mb = peak_mb(fn)
            except torch.cuda.OutOfMemoryError:
                mb = 'OOM'
            out['peak_mb'].setdefault(name, {})[f'B{B}_T{T}'] = mb
            print(f'  B={B:5d} T={T:3d}: {mb}', flush=True)
            del st, ac, y
            torch.cuda.empty_cache()
    json.dump(out, open('revision_experiments/results/train_peak_mem.json', 'w'), indent=2)
    print('\nSaved -> revision_experiments/results/train_peak_mem.json', flush=True)


if __name__ == '__main__':
    main()
