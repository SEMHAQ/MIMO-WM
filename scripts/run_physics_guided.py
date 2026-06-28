"""物理引导多尺度模型实验
每个分支有明确物理含义：
1. 位置预测分支（慢速SSM）- 预测位置/角度变化
2. 速度预测分支（快速SSM）- 预测速度/角速度变化
3. 外力估计分支（局部注意力）- 估计外力影响
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
T = 32

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

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

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params_m': round(params, 3), 'best_epoch': best_ep}

# ============================================================
# 物理引导多尺度模型
# ============================================================

class PhysicsGuidedMS(nn.Module):
    """物理引导多尺度模型
    - 位置分支：慢速SSM，预测位置/角度变化
    - 速度分支：快速SSM，预测速度/角速度变化
    - 外力分支：局部注意力，估计外力影响
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # 位置分支（慢速SSM）- 捕捉长程依赖
        self.position_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])

        # 速度分支（快速SSM）- 捕捉短程变化
        self.velocity_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)})
            for _ in range(n_layers)
        ])

        # 外力分支（局部注意力）- 估计瞬时外力
        self.force_branch = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)})
            for _ in range(n_layers)
        ])

        # 物理引导融合：三个分支分别预测不同物理量
        self.position_proj = nn.Linear(d_model, state_dim)  # 位置增量
        self.velocity_proj = nn.Linear(d_model, state_dim)  # 速度增量
        self.force_proj = nn.Linear(d_model, state_dim)  # 外力影响

        # 物理融合：位置 = 位置 + 速度 + 外力
        # 不需要额外参数，直接相加

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        # 位置分支
        pos_h = h
        for block in self.position_branch:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)

        # 速度分支
        vel_h = h
        for block in self.velocity_branch:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)

        # 外力分支
        force_h = h
        for block in self.force_branch:
            residual = force_h; x_norm = block['norm'](force_h); force_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)

        # 物理融合：位置 = 位置增量 + 速度增量 + 外力影响
        pos_delta = self.position_proj(pos_h[:, -1, :])
        vel_delta = self.velocity_proj(vel_h[:, -1, :])
        force_effect = self.force_proj(force_h[:, -1, :])

        # 最终预测 = 当前状态 + 位置增量 + 速度增量 + 外力影响
        return states[:, -1, :] + pos_delta + vel_delta + force_effect

# ============================================================
# 消融变体
# ============================================================

class NoForceBranch(nn.Module):
    """去掉外力分支"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.position_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)}) for _ in range(n_layers)])
        self.velocity_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)}) for _ in range(n_layers)])
        self.position_proj = nn.Linear(d_model, state_dim)
        self.velocity_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        pos_h = h
        for block in self.position_branch:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)
        vel_h = h
        for block in self.velocity_branch:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)
        pos_delta = self.position_proj(pos_h[:, -1, :])
        vel_delta = self.velocity_proj(vel_h[:, -1, :])
        return states[:, -1, :] + pos_delta + vel_delta

class NoVelocityBranch(nn.Module):
    """去掉速度分支"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.position_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)}) for _ in range(n_layers)])
        self.force_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)}) for _ in range(n_layers)])
        self.position_proj = nn.Linear(d_model, state_dim)
        self.force_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        pos_h = h
        for block in self.position_branch:
            residual = pos_h; x_norm = block['norm'](pos_h); pos_h = residual + block['ssm'](x_norm)
        force_h = h
        for block in self.force_branch:
            residual = force_h; x_norm = block['norm'](force_h); force_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)
        pos_delta = self.position_proj(pos_h[:, -1, :])
        force_effect = self.force_proj(force_h[:, -1, :])
        return states[:, -1, :] + pos_delta + force_effect

class NoPositionBranch(nn.Module):
    """去掉位置分支"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.velocity_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)}) for _ in range(n_layers)])
        self.force_branch = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)}) for _ in range(n_layers)])
        self.velocity_proj = nn.Linear(d_model, state_dim)
        self.force_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        vel_h = h
        for block in self.velocity_branch:
            residual = vel_h; x_norm = block['norm'](vel_h); vel_h = residual + block['ssm'](x_norm)
        force_h = h
        for block in self.force_branch:
            residual = force_h; x_norm = block['norm'](force_h); force_h = residual + block['conv'](x_norm.transpose(1,2)).transpose(1,2)
        vel_delta = self.velocity_proj(vel_h[:, -1, :])
        force_effect = self.force_proj(force_h[:, -1, :])
        return states[:, -1, :] + vel_delta + force_effect

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    # 加载所有数据集
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
        'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    }

    RESULTS_FILE = 'experiments/physics_guided_results.json'
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 定义消融配置
    configs = {
        '完整PG-MS': lambda sd, ad: PhysicsGuidedMS(sd, ad, d_model=96, d_state=8, n_layers=1, window_size=5),
        '去掉外力分支': lambda sd, ad: NoForceBranch(sd, ad, d_model=96, d_state=8, n_layers=1, window_size=5),
        '去掉速度分支': lambda sd, ad: NoVelocityBranch(sd, ad, d_model=96, d_state=8, n_layers=1, window_size=5),
        '去掉位置分支': lambda sd, ad: NoPositionBranch(sd, ad, d_model=96, d_state=8, n_layers=1, window_size=5),
    }

    for ds_name, ds_cfg in datasets.items():
        print(f'\n{"="*60}', flush=True)
        print(f'数据集: {ds_name}', flush=True)
        print(f'{"="*60}', flush=True)

        print(f'  加载数据...', flush=True)
        eps_tr = load_eps(ds_cfg['dir'], 'train')
        eps_vl = load_eps(ds_cfg['dir'], 'val')
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, T, m, s)
        Xv, Xav, Yv = make_data(eps_vl, T, m, s)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        for config_name, model_fn in configs.items():
            key = f'{config_name}_{ds_name}'
            if key in results and len(results[key]) >= len(SEEDS):
                print(f'\n{config_name}: 已有完整结果，跳过', flush=True)
                continue

            print(f'\n{config_name}:', flush=True)
            if key not in results:
                results[key] = {}

            for seed in SEEDS:
                seed_key = f'seed{seed}'
                if seed_key in results[key]:
                    print(f'  seed={seed} 已有，跳过', flush=True)
                    continue
                print(f'  seed={seed}...', end=' ', flush=True)
                model = model_fn(ds_cfg['sd'], ds_cfg['ad'])
                r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
                results[key][seed_key] = r
                mse_val = r['mse']
                r2_val = r['r2']
                print(f'MSE={mse_val:.4f}, R²={r2_val:.4f}', flush=True)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('物理引导多尺度模型结果', flush=True)
    print('='*60, flush=True)

    for ds_name in datasets:
        print(f'\n{ds_name}:', flush=True)
        print(f'{\"配置\":<20} {\"MSE(×10⁻²)\":<15} {\"R²\":<10} {\"参数(M)\":<10}')
        print('-'*60)
        for config_name in configs:
            key = f'{config_name}_{ds_name}'
            if key in results:
                valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
                if valid:
                    mses = [r['mse'] for r in valid]
                    r2s = [r['r2'] for r in valid]
                    params = valid[0]['params_m']
                    print(f'{config_name:<20} {np.mean(mses)*100:.2f}±{np.std(mses)*100:.2f}    {np.mean(r2s):.4f}    {params:.3f}')

    print('\nDone!', flush=True)
