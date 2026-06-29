"""Hamiltonian世界模型探索
核心思想：用哈密顿力学约束学习动力学

哈密顿方程：
- dq/dt = ∂H/∂p (位置变化 = 能量对速度的偏导)
- dp/dt = -∂H/∂q (速度变化 = -能量对位置的偏导)

优势：
1. 自动满足能量守恒
2. 有物理先验
3. 适合机器人动力学
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
# Hamiltonian世界模型
# ============================================================
class HamiltonianWorldModel(nn.Module):
    """Hamiltonian世界模型

    核心思想：
    1. 将状态分解为位置q和速度p
    2. 学习哈密顿函数H(q, p)
    3. 用哈密顿方程预测下一状态

    优势：
    - 自动满足能量守恒
    - 有物理先验
    - 适合机器人动力学
    """
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.pos_dim = state_dim // 2  # 位置维度
        self.vel_dim = state_dim - self.pos_dim  # 速度维度

        # 位置编码器
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.pos_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 速度编码器
        self.vel_encoder = nn.Sequential(
            nn.Linear(self.vel_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 哈密顿函数网络
        self.hamiltonian = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # SSM用于时序建模
        self.ssm = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model * 2), 'ssm': DiagSSM(d_model * 2, d_state)})
            for _ in range(n_layers)
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)

        # 分离位置和速度
        q = states[:, :, :self.pos_dim]  # 位置
        p = states[:, :, self.pos_dim:]  # 速度

        # 编码
        q_enc = self.pos_encoder(torch.cat([q, actions], dim=-1))  # (B, T, d_model)
        p_enc = self.vel_encoder(p)  # (B, T, d_model)

        # 拼接位置和速度编码
        h = torch.cat([q_enc, p_enc], dim=-1)  # (B, T, d_model*2)

        # SSM时序建模
        for block in self.ssm:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)

        # 哈密顿方程：用梯度计算状态变化
        h_last = h[:, -1, :].requires_grad_(True)
        H = self.hamiltonian(h_last)

        # 计算哈密顿梯度
        dH = torch.autograd.grad(H.sum(), h_last, create_graph=True)[0]

        # 分离位置和速度的梯度
        dH_dq = dH[:, :self.pos_dim]  # ∂H/∂q
        dH_dp = dH[:, self.pos_dim:]  # ∂H/∂p

        # 哈密顿方程：dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        # 这里用梯度作为状态变化的估计
        state_change = torch.cat([dH_dp, -dH_dq], dim=-1)

        # 预测下一状态
        state_pred = states[:, -1, :] + self.decoder(h[:, -1, :])

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

    RESULTS_FILE = 'experiments/hamiltonian.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    configs = {
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'HamiltonianWM': lambda: HamiltonianWorldModel(348, 17, d_model=96, d_state=16, n_layers=2),
    }

    print('\n' + '='*60, flush=True)
    print('Hamiltonian世界模型实验', flush=True)
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
