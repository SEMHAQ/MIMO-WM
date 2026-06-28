"""公平对比实验 - 所有模型用相同配置训练
确保：
1. 相同的数据预处理
2. 相同的训练参数
3. 相同的随机种子
4. 相同的评估方法
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
LR = 5e-4
T = 32

print(f'Device: {device}', flush=True)

# ============================================================
# 数据加载（统一）
# ============================================================
def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    eps = []
    for i, f in enumerate(fs):
        eps.append((np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']))
        if (i+1) % 200 == 0: print(f'    {i+1}/{len(fs)}...', flush=True)
    print(f'    {len(fs)} episodes loaded', flush=True)
    return eps

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
# 模型定义（统一）
# ============================================================

class SimpleSSM(nn.Module):
    """简单SSM（S4D风格，无gate无FFN）"""
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

class MultiScaleModel(nn.Module):
    """MS-WM（多尺度融合）"""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=8, n_layers=1, window_size=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.slow_ssm = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)}) for _ in range(n_layers)])
        self.fast_ssm = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state // 2)}) for _ in range(n_layers)])
        self.local_attn = nn.ModuleList([nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'conv': nn.Conv1d(d_model, d_model, kernel_size=window_size, padding=window_size//2, groups=d_model)}) for _ in range(n_layers)])
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 3, 3), nn.Softmax(dim=-1))
        self.fusion_proj = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        slow_h = h
        for b in self.slow_ssm: residual = slow_h; slow_h = residual + b['ssm'](b['norm'](slow_h))
        fast_h = h
        for b in self.fast_ssm: residual = fast_h; fast_h = residual + b['ssm'](b['norm'](fast_h))
        local_h = h
        for b in self.local_attn: residual = local_h; local_h = residual + b['conv'](b['norm'](local_h).transpose(1,2)).transpose(1,2)
        features = torch.cat([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=-1)
        gate = self.fusion_gate(features)
        stacked = torch.stack([slow_h[:, -1, :], fast_h[:, -1, :], local_h[:, -1, :]], dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        return states[:, -1, :] + self.fusion_proj(fused)

# ============================================================
# 训练函数（统一）
# ============================================================
def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
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
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
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

    # 测量推理时间
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000

    return {
        'mse': round(mse, 6),
        'r2': round(r2, 4),
        'params_m': round(params, 3),
        'inf_time_ms': round(inf_time, 2),
        'best_epoch': best_ep
    }

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    # 数据集配置
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
        'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    }

    # 模型配置（统一d_model=96, d_state=16, n_layers=2）
    models = {
        'LSTM-WM': (LSTMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 96, 'n_layers': 2}),
        'GRU-WM': (GRUWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 96, 'n_layers': 2}),
        'Transformer-WM': (TransformerWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'nhead': 4, 'n_layers': 2}),
        'Mamba-WM': (MambaWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'n_layers': 2}),
        'S4D-WM': (SSMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
        'SSM-WM': (SimpleSSM, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
        'MS-WM': (MultiScaleModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 96, 'd_state': 8, 'n_layers': 1, 'window_size': 5}),
    }

    RESULTS_FILE = 'experiments/fair_comparison.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

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

        for model_name, (ModelClass, kwargs_fn) in models.items():
            key = f'{model_name}_{ds_name}'
            if key in results and len(results[key]) >= len(SEEDS):
                print(f'\n{model_name}: 已有完整结果，跳过', flush=True)
                continue

            print(f'\n{model_name}:', flush=True)
            if key not in results:
                results[key] = {}

            kwargs = kwargs_fn(ds_cfg['sd'], ds_cfg['ad'])
            for seed in SEEDS:
                seed_key = f'seed{seed}'
                if seed_key in results[key]:
                    print(f'  seed={seed} 已有，跳过', flush=True)
                    continue
                print(f'  seed={seed}...', end=' ', flush=True)
                r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
                results[key][seed_key] = r
                mse_val = r['mse']
                r2_val = r['r2']
                print(f'MSE={mse_val:.4f}, R²={r2_val:.4f}', flush=True)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    # 打印结果汇总
    print('\n' + '='*60, flush=True)
    print('公平对比结果汇总', flush=True)
    print('='*60, flush=True)

    for ds_name in datasets:
        print(f'\n{ds_name}:', flush=True)
        print('{:<16} {:<15} {:<10} {:<10} {:<10}'.format('模型', 'MSE(×10⁻²)', 'R²', '参数(M)', '推理(ms)'))
        print('-'*65)
        for model_name in models:
            key = f'{model_name}_{ds_name}'
            if key in results:
                valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
                if valid:
                    mses = [r['mse'] for r in valid]
                    r2s = [r['r2'] for r in valid]
                    params = valid[0]['params_m']
                    inf_time = valid[0]['inf_time_ms']
                    print('{:<16} {:.2f}±{:.2f}    {:.4f}    {:.3f}    {:.2f}'.format(
                        model_name, np.mean(mses)*100, np.std(mses)*100, np.mean(r2s), params, inf_time
                    ))

    print('\nDone!', flush=True)
