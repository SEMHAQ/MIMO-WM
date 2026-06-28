"""Train MS-WM for each T value on Ant dataset."""
import torch, torch.nn as nn, numpy as np, sys, os, json
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MS-WM模型（与run_all_experiments.py一致）
class MultiScaleModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5, fusion_type='gate'):
        super().__init__()
        self.state_dim = state_dim
        self.fusion_type = fusion_type
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.slow_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.fast_ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])
        self.local_attn = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)})
            for _ in range(n_layers)
        ])
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 3, 3), nn.Softmax(dim=-1))
        self.fusion_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        slow_h = h
        for block in self.slow_ssm:
            residual = slow_h; x_norm = block['norm'](slow_h); slow_h = residual + block['ssm'](x_norm)
        fast_h = h
        for block in self.fast_ssm:
            residual = fast_h; x_norm = block['norm'](fast_h); fast_h = residual + block['ssm'](x_norm)
        local_h = h
        for block in self.local_attn:
            residual = local_h; x_norm = block['norm'](local_h); local_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)
        features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
        gate = self.fusion_gate(features)
        stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        pred = self.fusion_proj(fused)
        return states[:, -1, :] + pred

def load_episodes(data_dir, split):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def make_step_data(episodes, T, mean=None, std=None):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        if mean is not None: st = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def compute_stats(episodes):
    all_states = np.concatenate([st for st, _ in episodes], axis=0)
    return all_states.mean(axis=0), all_states.std(axis=0)

# 只跑Ant
dataset = 'ant'
state_dim = 105
action_dim = 8
T_values = [16, 32, 64, 128, 256]
SEEDS = [42, 123, 456, 789, 1024]

print(f'加载Ant数据...', flush=True)
eps_tr = load_episodes(f'data/{dataset}', 'train')
eps_vl = load_episodes(f'data/{dataset}', 'val')
mean, std = compute_stats(eps_tr)

RESULTS_FILE = f'experiments/{dataset}_seqlen_results.json'
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
else:
    results = {}

print(f'\n{"="*60}\nAnt序列长度实验 (MS-WM)\n{"="*60}', flush=True)

for T in T_values:
    key = f'T{T}'
    if key in results and len(results[key]) >= len(SEEDS):
        print(f'T={T}: 已有完整结果，跳过', flush=True)
        continue

    print(f'T={T}:', flush=True)
    Xs, Xa, Y = make_step_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_step_data(eps_vl, T, mean, std)
    if len(Xs) == 0 or len(Xv) == 0:
        print(f'  数据不足，跳过', flush=True)
        continue

    if key not in results:
        results[key] = {}

    for seed in SEEDS:
        seed_key = f'seed{seed}'
        if seed_key in results[key]:
            print(f'  seed={seed} 已有，跳过', flush=True)
            continue

        print(f'  seed={seed}...', end=' ', flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = MultiScaleModel(state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        loss_fn = nn.MSELoss()
        Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
        best_val = float('inf'); pat = 0
        for ep in range(100):
            model.train()
            idx = np.random.permutation(len(Xs))
            for i in range(0, len(idx), 256):
                bi = idx[i:i+256]
                pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
                loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sch.step()
            model.eval()
            with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
            if vl < best_val: best_val = vl; pat = 0
            else: pat += 1
            if pat >= 20: break
        model.eval()
        with torch.no_grad():
            pred = model(Xv_g, Xav_g)
            mse = loss_fn(pred, Yv_g).item()
            ss_r = torch.sum((Yv_g - pred)**2).item()
            ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
            r2 = 1 - ss_r / ss_t
        results[key][seed_key] = {'mse': round(mse, 6), 'r2': round(r2, 4)}
        print(f'MSE={mse:.4f} R²={r2:.4f}', flush=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

# 打印结果
print(f'\n{"="*60}\n结果汇总\n{"="*60}', flush=True)
print(f'{"T":<6} {"MSE(×10⁻²)":<15} {"R²":<10}', flush=True)
print('-'*35, flush=True)
for T in T_values:
    key = f'T{T}'
    if key in results:
        valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
        if valid:
            mses = [r['mse'] for r in valid]
            r2s = [r['r2'] for r in valid]
            print(f'{T:<6} {np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}    {np.mean(r2s):.4f}', flush=True)

print('\nDone!', flush=True)
