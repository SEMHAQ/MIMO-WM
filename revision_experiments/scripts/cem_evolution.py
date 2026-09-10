"""A9: CEM-MPC 代价下降 + 采样方差演化 (真实非线性代价下) —— 多次随机规划问题.
用途:展示 μ-强凸假设不满足时 CEM 的实际收敛行为与早熟/方差演化.

依赖 Wave-1 保存的 MIMO-WM Humanoid seed42 权重; GPU 空闲时运行:
  wsl -e bash -lc "cd /mnt/e/Project/SSM-World-Model && python3 revision_experiments/scripts/cem_evolution.py"
产物: revision_experiments/results/cem_evolution.json
"""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel

SD, AD, T = 348, 17, 32
CKPT = 'revision_experiments/results/ckpt/MIMO-WM_humanoid_seed42.pt'
H, K, NE, M, NPROB = 6, 256, 32, 12, 5  # 演化用较大 M; 与正文 CEM(K256,Ne32,M5)同量级


def load_eps(d, s):
    import glob
    fs = sorted(glob.glob(os.path.join(d, s, '*.npz')))[:40]
    return [(np.load(f)['states'], np.load(f)['actions']) for f in fs]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MIMOWorldModel(SD, AD, d_model=96, d_state=16, n_layers=2).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    eps = load_eps('data/humanoid', 'val')
    a = np.concatenate([s for s, _ in eps]); mean, std = a.mean(0), a.std(0)

    prob = []
    for states, actions in eps:
        if len(states) < T + H + 5:
            continue
        sn = (states - mean) / (std + 1e-8)
        s_hist = sn[:T]
        a_hist = actions[:T - 1]
        tgt = sn[T + 2]  # 一个有代表性的远期参考
        prob.append((s_hist, a_hist, tgt))
        if len(prob) >= NPROB:
            break

    all_iter = []
    for pidx, (s_hist, a_hist, tgt) in enumerate(prob):
        s_h = torch.FloatTensor(s_hist).unsqueeze(0).to(dev)
        a_h = torch.FloatTensor(a_hist).unsqueeze(0).to(dev)
        tgt_g = torch.FloatTensor(tgt).to(dev)
        adim = a_hist.shape[1]
        mu = torch.zeros(H, adim, device=dev)
        sigma = torch.ones(H, adim, device=dev)
        trace = []
        with torch.no_grad():
            for it in range(M):
                samples = (mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(K, H, adim, device=dev)).clamp(-1, 1)
                costs = torch.zeros(K, device=dev)
                s_cur = s_h.expand(K, -1, -1).clone()
                a_cur = a_h.expand(K, -1, -1).clone()
                for h in range(H):
                    pred = model(s_cur, a_cur)
                    costs += torch.norm(pred - tgt_g.unsqueeze(0), p=2, dim=-1) + 0.01 * torch.norm(samples[:, h], p=2, dim=-1)
                    s_cur = torch.cat([s_cur[:, 1:], pred.unsqueeze(1)], dim=1)
                    a_cur = torch.cat([a_cur[:, 1:], samples[:, h:h + 1]], dim=1)
                elite = samples[costs.topk(NE, largest=False).indices]
                elite_cost = costs.topk(NE, largest=False).values.mean().item()
                mu = elite.mean(0)
                sigma = elite.std(0) + 1e-6
                trace.append({'iter': it + 1, 'elite_mean_cost': round(elite_cost, 4),
                              'sigma_fro': round(sigma.norm().item(), 4)})
        all_iter.append({'problem': pidx, 'trace': trace})
        print(f'prob {pidx}: elite_cost {trace[0]["elite_mean_cost"]} -> {trace[-1]["elite_mean_cost"]}, '
              f'sigma_fro {trace[0]["sigma_fro"]} -> {trace[-1]["sigma_fro"]}', flush=True)
    json.dump({'H': H, 'K': K, 'NE': NE, 'M': M, 'problems': all_iter},
              open('revision_experiments/results/cem_evolution.json', 'w'), indent=2)
    print('Saved -> revision_experiments/results/cem_evolution.json', flush=True)


if __name__ == '__main__':
    main()
