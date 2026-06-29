"""多方向探索 - 测试多种创新架构

方向：
1. Neural ODE + SSM
2. Transformer + 物理约束
3. 集成模型
4. 简单SSM（基线）
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
# 方向1: Neural ODE风格SSM
# ============================================================
class NeuralODE_SSM(nn.Module):
    """Neural ODE风格：用SSM学习连续时间动力学"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])
        # 残差连接：学习状态变化量
        self.delta_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)
        # 预测状态变化量（类似ODE）
        delta = self.delta_net(h[:, -1, :])
        return states[:, -1, :] + delta

# ============================================================
# 方向2: 物理约束Transformer
# ============================================================
class PhysicsTransformer(nn.Module):
    """物理约束Transformer：用注意力机制 + 物理先验"""
    def __init__(self, state_dim, action_dim, d_model=96, nhead=4, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        # 因果掩码
        seq_len = h.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

# ============================================================
# 方向3: 集成模型
# ============================================================
class EnsembleSSM(nn.Module):
    """集成SSM：多个独立SSM，平均预测"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2, n_ensemble=3):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                DiagSSM(d_model, d_state),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, state_dim)
            ) for _ in range(n_ensemble)
        ])

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        # 集成预测
        preds = []
        for model in self.models:
            h = model[:4](x)  # encoder + norm
            h = model[4](h)   # SSM
            pred = states[:, -1, :] + model[5:](h[:, -1, :])  # decoder
            preds.append(pred)
        return torch.stack(preds).mean(dim=0)

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

    RESULTS_FILE = 'experiments/all_directions.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    configs = {
        'NeuralODE-SSM': lambda: NeuralODE_SSM(348, 17, d_model=96, d_state=16, n_layers=2),
        'PhysicsTransformer': lambda: PhysicsTransformer(348, 17, d_model=96, nhead=4, n_layers=2),
        'EnsembleSSM': lambda: EnsembleSSM(348, 17, d_model=96, d_state=16, n_layers=2, n_ensemble=3),
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=96, d_state=16, n_layers=2),
    }

    print('\n' + '='*60, flush=True)
    print('多方向探索实验', flush=True)
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
    print('{:<25} {:<10} {:<10} {:<10}'.format('模型', 'MSE', 'R²', '参数(M)'))
    print('-'*55)

    for name in configs:
        if name in results:
            r = results[name]
            print('{:<25} {:<10.4f} {:<10.4f} {:<10.3f}'.format(name, r['mse'], r['r2'], r['params_m']))

    print('\nDone!', flush=True)
