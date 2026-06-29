"""SSM-TD-MPC2: 用SSM替换TD-MPC2的MLP动力学模型

借鉴TD-MPC2的完整架构：
1. 编码器：状态 → 隐空间
2. SSM动力学：在隐空间做SSM预测（替换MLP）
3. 奖励预测
4. 策略先验
5. Q函数集成
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
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

# ============================================================
# 基线：简单SSM
# ============================================================
class SimpleSSM(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# SSM-TD-MPC2 世界模型
# ============================================================
class SSM_TD_MPC2(nn.Module):
    """SSM-TD-MPC2: 用SSM替换TD-MPC2的MLP动力学模型

    组件：
    1. 编码器：状态 → 隐空间
    2. SSM动力学：在隐空间做SSM预测
    3. 奖励预测
    4. 策略先验
    5. Q函数集成
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2, latent_dim=64, num_q=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_q = num_q

        # 编码器：状态 → 隐空间
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )

        # SSM动力学模型（在隐空间）
        self.dynamics = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(latent_dim + action_dim), 'ssm': DiagSSM(latent_dim + action_dim, d_state)})
            for _ in range(n_layers)
        ])
        self.dynamics_proj = nn.Linear(latent_dim + action_dim, latent_dim)

        # 奖励预测
        self.reward = nn.Sequential(
            nn.Linear(latent_dim + action_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # 策略先验
        self.pi = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * action_dim)  # mean + log_std
        )

        # Q函数集成
        self.Qs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            ) for _ in range(num_q)
        ])

    def encode(self, states):
        """编码状态到隐空间"""
        return self.encoder(states)

    def next_latent(self, z, actions):
        """预测下一隐状态"""
        x = torch.cat([z, actions], dim=-1)
        h = x
        for block in self.dynamics:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)
        return self.dynamics_proj(h[:, -1, :])

    def predict_reward(self, z, actions):
        """预测奖励"""
        x = torch.cat([z, actions], dim=-1)
        return self.reward(x[:, -1, :])

    def predict_action(self, z):
        """策略先验：预测动作分布"""
        out = self.pi(z[:, -1, :])
        mean, log_std = out.chunk(2, dim=-1)
        return mean, log_std

    def predict_q(self, z, actions):
        """Q函数集成"""
        x = torch.cat([z, actions], dim=-1)
        q_values = [q(x[:, -1, :]) for q in self.Qs]
        return torch.stack(q_values, dim=0).min(dim=0)[0]

    def forward(self, states, actions):
        """前向传播：预测下一状态"""
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)

        # 编码
        z = self.encode(states)  # (B, T, latent_dim)

        # 隐空间动力学
        z_next = self.next_latent(z, actions)  # (B, latent_dim)

        # 解码回状态空间（简化：直接用线性层）
        state_pred = states[:, -1, :] + nn.Linear(self.latent_dim, states.shape[-1]).to(device)(z_next)

        return state_pred

# ============================================================
# 训练函数
# ============================================================
def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
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
# 主实验
# ============================================================
if __name__ == '__main__':
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    RESULTS_FILE = 'experiments/ssm_tdmpc.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    configs = {
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'SSM-TD-MPC2-d32': lambda: SSM_TD_MPC2(348, 17, d_model=96, d_state=16, n_layers=2, latent_dim=32),
        'SSM-TD-MPC2-d64': lambda: SSM_TD_MPC2(348, 17, d_model=96, d_state=16, n_layers=2, latent_dim=64),
        'SSM-TD-MPC2-d128': lambda: SSM_TD_MPC2(348, 17, d_model=96, d_state=16, n_layers=2, latent_dim=128),
    }

    print('\n' + '='*60, flush=True)
    print('SSM-TD-MPC2实验', flush=True)
    print('='*60, flush=True)

    for name, model_fn in configs.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        model = model_fn()
        r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv)
        results[name] = r
        print(f'  MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params_m"]:.3f}M', flush=True)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('结果汇总', flush=True)
    print('='*60, flush=True)
    print('{:<20} {:<10} {:<10} {:<10}'.format('模型', 'MSE', 'R²', '参数(M)'))
    print('-'*50)

    for name in configs:
        if name in results:
            r = results[name]
            print('{:<20} {:<10.4f} {:<10.4f} {:<10.3f}'.format(name, r['mse'], r['r2'], r['params_m']))

    print('\nDone!', flush=True)
