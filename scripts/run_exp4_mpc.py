"""实验4：MPC规划 — 6个世界模型 × 2种MPC方法

世界模型：LSTM, GRU, Transformer, Mamba, S4D, MIMO-WM
MPC方法：梯度MPC (Adam), CEM-MPC (GPU并行采样)
数据集：Humanoid
指标：控制频率(Hz)、每步耗时(ms)
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel
from src.models.mamba_world_model import MambaWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 32
SEEDS = [42]

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs[:50]]

def stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

def make_data(eps, T, mean, std):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

# ============================================================
# MIMO-WM 模型（与exp1一致）
# ============================================================
class MIMOLayer(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.output(x) * torch.sigmoid(self.gate(x))
        return residual + x

class MIMOWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# 梯度MPC
# ============================================================
class GradientMPC:
    def __init__(self, world_model, horizon=10, n_iterations=30, lr=0.01, Q=1.0, R=0.01):
        self.model = world_model
        self.H = horizon
        self.n_iter = n_iterations
        self.lr = lr
        self.Q = Q
        self.R = R

    def plan(self, state_hist, action_hist, target):
        action_dim = action_hist.shape[1]
        action_seq = nn.Parameter(torch.randn(self.H, action_dim, device=device) * 0.1)
        opt = torch.optim.Adam([action_seq], lr=self.lr)
        s_hist = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_hist = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)

        was_eval = not self.model.training
        if was_eval: self.model.train()

        for _ in range(self.n_iter):
            opt.zero_grad()
            cost = torch.tensor(0.0, device=device)
            s, a = s_hist.clone(), a_hist.clone()
            for h in range(self.H):
                pred = self.model(s, a)
                cost = cost + self.Q * torch.norm(pred - tgt, p=2) + self.R * torch.norm(action_seq[h], p=2)
                s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
                a = torch.cat([a[:, 1:], action_seq[h:h+1].unsqueeze(0)], dim=1)
            cost.backward(); opt.step()

        if was_eval: self.model.eval()
        return action_seq[0].detach().cpu().numpy()

# ============================================================
# CEM-MPC
# ============================================================
class CEMMPC:
    def __init__(self, world_model, horizon=10, n_samples=256, n_elite=32, n_iterations=5, Q=1.0, R=0.01):
        self.model = world_model
        self.H = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iter = n_iterations
        self.Q = Q
        self.R = R

    def plan(self, state_hist, action_hist, target):
        action_dim = action_hist.shape[1]
        s_hist = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_hist = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)
        mu = torch.zeros(self.H, action_dim, device=device)
        sigma = torch.ones(self.H, action_dim, device=device)

        for _ in range(self.n_iter):
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(self.n_samples, self.H, action_dim, device=device)
            samples = samples.clamp(-1, 1)
            costs = torch.zeros(self.n_samples, device=device)
            s_cur = s_hist.expand(self.n_samples, -1, -1).clone()
            a_cur = a_hist.expand(self.n_samples, -1, -1).clone()
            for h in range(self.H):
                pred = self.model(s_cur, a_cur)
                costs = costs + self.Q * torch.norm(pred - tgt.unsqueeze(0), p=2, dim=-1) + self.R * torch.norm(samples[:, h], p=2, dim=-1)
                s_cur = torch.cat([s_cur[:, 1:], pred.unsqueeze(1)], dim=1)
                a_cur = torch.cat([a_cur[:, 1:], samples[:, h:h+1]], dim=1)
            elite_idx = costs.topk(self.n_elite, largest=False).indices
            elite = samples[elite_idx]
            mu = elite.mean(dim=0)
            sigma = elite.std(dim=0) + 1e-6

        return mu[0].detach().cpu().numpy()

# ============================================================
# 训练世界模型
# ============================================================
def train_world_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0
    for ep in range(100):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    return model

# ============================================================
# MPC速度评估
# ============================================================
def evaluate_mpc_speed(mpc, episodes, mean, std, n_episodes=10, n_steps=50):
    total_time = 0; count = 0
    for states, actions in episodes[:n_episodes]:
        if len(states) < T + n_steps: continue
        s_norm = (states - mean) / (std + 1e-8)
        target = s_norm[T]
        t0 = time.perf_counter()
        for step in range(n_steps):
            s_hist = s_norm[step:step+T]
            a_hist = actions[step:step+T-1]
            mpc.plan(s_hist, a_hist, target)
            if T + step + 1 < len(s_norm):
                target = s_norm[T + step + 1]
        elapsed = time.perf_counter() - t0
        total_time += elapsed; count += 1
    if count == 0: return {'hz': 0, 'avg_step_time_ms': 0, 'episodes': 0}
    avg_time = total_time / count / n_steps
    return {
        'hz': round(1.0 / avg_time, 2) if avg_time > 0 else 0,
        'avg_step_time_ms': round(avg_time * 1000, 2),
        'episodes': count
    }

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    ds_cfg = {'dir': 'data/humanoid', 'sd': 348, 'ad': 17}

    models = {
        'LSTM-WM':        (LSTMWorldModel,       {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
        'GRU-WM':         (GRUWorldModel,         {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
        'Transformer-WM': (TransformerWorldModel,  {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'nhead': 4, 'n_layers': 2}),
        'Mamba-WM':       (MambaWorldModel,        {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'n_layers': 2}),
        'S4D-WM':         (SSMWorldModel,          {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
        'MIMO-WM':        (MIMOWorldModel,         {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
    }

    RESULTS_FILE = 'experiments/exp4_mpc.json'
    os.makedirs('experiments', exist_ok=True)
    results = {}

    eps_tr = load_eps(ds_cfg['dir'], 'train')
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    mean, std = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_data(eps_vl, T, mean, std)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    for model_name, (ModelClass, kwargs) in models.items():
        print(f'\n[{model_name}]', flush=True)
        print(f'  训练模型...', flush=True)
        model = train_world_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv)
        model.eval()

        print(f'  评估梯度MPC...', flush=True)
        grad_mpc = GradientMPC(model, horizon=10, n_iterations=30, lr=0.01)
        r = evaluate_mpc_speed(grad_mpc, eps_vl, mean, std)
        results[f'{model_name}_GradMPC'] = r
        print(f'    {r["hz"]:.2f} Hz ({r["avg_step_time_ms"]:.1f} ms)', flush=True)

        print(f'  评估CEM-MPC...', flush=True)
        cem_mpc = CEMMPC(model, horizon=10, n_samples=256, n_elite=32, n_iterations=5)
        r = evaluate_mpc_speed(cem_mpc, eps_vl, mean, std)
        results[f'{model_name}_CEMMPC'] = r
        print(f'    {r["hz"]:.2f} Hz ({r["avg_step_time_ms"]:.1f} ms)', flush=True)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 汇总
    print('\n' + '='*75, flush=True)
    print('MPC控制频率对比 (Hz)', flush=True)
    print('='*75)
    print('{:<18} {:<12} {:<12} {:<10}'.format('模型', '梯度MPC', 'CEM-MPC', '加速比'))
    print('-'*75)
    for m in models:
        g = results[f'{m}_GradMPC']['hz']
        c = results[f'{m}_CEMMPC']['hz']
        print('{:<18} {:<12.2f} {:<12.2f} {:.1f}x'.format(m, g, c, c/g if g>0 else 0))
    print('\nDone!', flush=True)
