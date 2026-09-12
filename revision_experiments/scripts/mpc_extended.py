"""MPC 扩展对比：把返修新增的 5 个基线也纳入 MPC 口径（Humanoid）。
训练配方与 run_exp4_mpc.py 一致（BS=1024, 100 epochs, lr=5e-4, 早停 pat=20），
评测：梯度 MPC 频率 / CEM-MPC 频率 / 3 步预测 MSE。
结果增量写入 revision_experiments/results/mpc_extended.json（可断点续跑）。

运行（WSL GPU）:
  python3 -u revision_experiments/scripts/mpc_extended.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))
from src.models.baselines import TransformerWorldModel
from src.models.ssm_world_model import SSMWorldModel
from revision_experiments.scripts.run_revision_matrix import (
    NoGateMIMO, LRUWorldModel, PerformerWorldModel)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 32
EPOCHS, BS, LR = 100, 1024, 5e-4
RESULT = 'revision_experiments/results/mpc_extended.json'
os.makedirs('revision_experiments/results', exist_ok=True)
print('Device:', DEVICE, flush=True)

SD, AD = 348, 17
MODELS = {
    'MIMO-SSM-noGate': (NoGateMIMO,      dict(state_dim=SD, action_dim=AD, d_model=96, d_state=16, n_layers=2)),
    'S4D-WM':          (SSMWorldModel,   dict(state_dim=SD, action_dim=AD, d_model=96, d_state=16, n_layers=2)),
    'LRU-WM':          (LRUWorldModel,   dict(state_dim=SD, action_dim=AD, d_model=96, d_state=16, n_layers=2)),
    'Performer-WM':    (PerformerWorldModel, dict(state_dim=SD, action_dim=AD, d_model=96, n_layers=2, feat=192)),
    'Transformer-Reg': (TransformerWorldModel, dict(state_dim=SD, action_dim=AD, d_model=192, nhead=6, n_layers=3)),
}


def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]


def make_data(eps, mean, std):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T + 1:
            continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st) - T, T):
            if j + T >= len(st):
                break
            Xs.append(sn[j:j + T]); Xa.append(ac[j:j + T - 1]); Y.append(sn[j + T])
    return np.array(Xs), np.array(Xa), np.array(Y)


def train_model(ModelClass, kw, Xs, Xa, Y, Xv, Xav, Yv, seed=42, bs=BS):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ModelClass(**kw).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(DEVICE); Xav_g = torch.FloatTensor(Xav).to(DEVICE); Yv_g = torch.FloatTensor(Yv).to(DEVICE)
    best, pat = float('inf'), 0; best_sd = None

    def val_loss():
        tot, n = 0.0, len(Xv_g)
        with torch.no_grad():
            for i in range(0, n, 1024):
                s = Xv_g[i:i + 1024]
                tot += fn(m(s, Xav_g[i:i + 1024]), Yv_g[i:i + 1024]).item() * s.shape[0]
        return tot / n

    for ep in range(EPOCHS):
        m.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), bs):
            bi = idx[i:i + bs]
            loss = fn(m(torch.FloatTensor(Xs[bi]).to(DEVICE), torch.FloatTensor(Xa[bi]).to(DEVICE)),
                      torch.FloatTensor(Y[bi]).to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        vl = val_loss()
        if vl < best:
            best, pat = vl, 0; best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            pat += 1
        if pat >= 20:
            break
    m.load_state_dict(best_sd)
    return m


class GradientMPC:
    def __init__(self, model, horizon=10, n_iter=30, lr=0.01, Q=1.0, R=0.01):
        self.m = model; self.H = horizon; self.n_iter = n_iter; self.lr = lr; self.Q = Q; self.R = R

    def plan(self, sh, ah, tgt):
        a_dim = ah.shape[1]
        a_seq = nn.Parameter(torch.randn(self.H, a_dim, device=DEVICE) * 0.1)
        opt = torch.optim.Adam([a_seq], lr=self.lr)
        s_h = torch.FloatTensor(sh).unsqueeze(0).to(DEVICE)
        a_h = torch.FloatTensor(ah).unsqueeze(0).to(DEVICE)
        tgt = torch.FloatTensor(tgt).to(DEVICE)
        was_eval = not self.m.training
        if was_eval:
            self.m.train()
        for _ in range(self.n_iter):
            opt.zero_grad()
            cost = torch.tensor(0.0, device=DEVICE)
            s, a = s_h.clone(), a_h.clone()
            for h in range(self.H):
                pred = self.m(s, a)
                cost = cost + self.Q * torch.norm(pred - tgt) + self.R * torch.norm(a_seq[h])
                s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
                a = torch.cat([a[:, 1:], a_seq[h:h + 1].unsqueeze(0)], dim=1)
            cost.backward(); opt.step()
        if was_eval:
            self.m.eval()
        return a_seq[0].detach().cpu().numpy()


class CEMMPC:
    def __init__(self, model, horizon=10, K=256, Ne=32, M=5, Q=1.0, R=0.01):
        self.m = model; self.H = horizon; self.K = K; self.Ne = Ne; self.M = M; self.Q = Q; self.R = R

    def plan(self, sh, ah, tgt):
        a_dim = ah.shape[1]
        s_h = torch.FloatTensor(sh).unsqueeze(0).to(DEVICE)
        a_h = torch.FloatTensor(ah).unsqueeze(0).to(DEVICE)
        tgt = torch.FloatTensor(tgt).to(DEVICE)
        mu = torch.zeros(self.H, a_dim, device=DEVICE)
        sigma = torch.ones(self.H, a_dim, device=DEVICE)
        for _ in range(self.M):
            samples = (mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(self.K, self.H, a_dim, device=DEVICE)).clamp(-1, 1)
            costs = torch.zeros(self.K, device=DEVICE)
            s_cur = s_h.expand(self.K, -1, -1).clone(); a_cur = a_h.expand(self.K, -1, -1).clone()
            for h in range(self.H):
                pred = self.m(s_cur, a_cur)
                costs = costs + self.Q * torch.norm(pred - tgt.unsqueeze(0), p=2, dim=-1) + self.R * torch.norm(samples[:, h], p=2, dim=-1)
                s_cur = torch.cat([s_cur[:, 1:], pred.unsqueeze(1)], dim=1)
                a_cur = torch.cat([a_cur[:, 1:], samples[:, h:h + 1]], dim=1)
            elite = samples[costs.topk(self.Ne, largest=False).indices]
            mu = elite.mean(0); sigma = elite.std(0) + 1e-6
        return mu[0].detach().cpu().numpy()


def eval_speed(mpc, eps_vl, mean, std, n_ep=10, n_steps=50):
    total, cnt = 0.0, 0
    for states, actions in eps_vl[:n_ep]:
        if len(states) < T + n_steps:
            continue
        s_norm = (states - mean) / (std + 1e-8)
        target = s_norm[T]
        t0 = time.perf_counter()
        for step in range(n_steps):
            mpc.plan(s_norm[step:step + T], actions[step:step + T - 1], target)
            if T + step + 1 < len(s_norm):
                target = s_norm[T + step + 1]
        total += time.perf_counter() - t0; cnt += 1
    if cnt == 0:
        return {'hz': 0, 'avg_step_time_ms': 0}
    avg = total / cnt / n_steps
    return {'hz': round(1.0 / avg, 2) if avg > 0 else 0, 'avg_step_time_ms': round(avg * 1000, 2)}


def eval_quality(model, eps_vl, mean, std, n_ep=5, H=3):
    model.eval()
    errs = []
    for states, actions in eps_vl[:n_ep]:
        if len(states) < T + H:
            continue
        s_norm = (states - mean) / (std + 1e-8)
        s_cur = s_norm[:T].copy(); a_cur = actions[:T - 1].copy()
        real = s_norm[T:T + H]; fut_a = actions[T - 1:T - 1 + H]
        preds = []
        for h in range(H):
            with torch.no_grad():
                p = model(torch.FloatTensor(s_cur).unsqueeze(0).to(DEVICE),
                          torch.FloatTensor(a_cur).unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()
            preds.append(p)
            s_cur = np.vstack([s_cur[1:], p.reshape(1, -1)])
            a_cur = np.vstack([a_cur[1:], fut_a[h].reshape(1, -1)])
        errs.append(float(np.mean(np.mean((np.array(preds) - real) ** 2, axis=1))))
    return round(float(np.mean(errs)), 6) if errs else 0


if __name__ == '__main__':
    eps_tr = load_eps('data/humanoid', 'train'); eps_vl = load_eps('data/humanoid', 'val')
    a = np.concatenate([s for s, _ in eps_tr]); mean, std = a.mean(0), a.std(0)
    Xs, Xa, Y = make_data(eps_tr, mean, std)
    Xv, Xav, Yv = make_data(eps_vl, mean, std)
    print(f'Train {len(Xs)} Val {len(Xv)}', flush=True)
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

    for name, (MC, kw) in MODELS.items():
        gk, ck, qk = f'{name}_GradMPC', f'{name}_CEMMPC', f'{name}_Quality'
        if all(k in res for k in (gk, ck, qk)):
            print(f'{name}: 已有结果, 跳过', flush=True); continue
        bs_use = 256 if name == 'Performer-WM' else BS   # Performer 的随机特征展开显存开销大
        print(f'\n[{name}] 训练... (bs={bs_use})', flush=True)
        m = train_model(MC, kw, Xs, Xa, Y, Xv, Xav, Yv, bs=bs_use)
        if gk not in res:
            print('  梯度 MPC...', flush=True); res[gk] = eval_speed(GradientMPC(m), eps_vl, mean, std)
            print('   ', res[gk], flush=True)
        if ck not in res:
            print('  CEM MPC...', flush=True); res[ck] = eval_speed(CEMMPC(m), eps_vl, mean, std)
            print('   ', res[ck], flush=True)
        if qk not in res:
            res[qk] = eval_quality(m, eps_vl, mean, std)
            print('   3步 MSE =', res[qk], flush=True)
        json.dump(res, open(RESULT, 'w'), indent=2)

    print('\n=== MPC 扩展结果 ===', flush=True)
    for name in MODELS:
        g = res.get(f'{name}_GradMPC', {}).get('hz', 0)
        c = res.get(f'{name}_CEMMPC', {}).get('hz', 0)
        q = res.get(f'{name}_Quality', 0)
        print(f'{name:<18} Grad {g:>6} Hz | CEM {c:>6} Hz | 3步MSE {q}', flush=True)
    print('\nDone!', flush=True)
