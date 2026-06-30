"""MIMO-WM 消融实验 v2

消融维度：
A. 核心组件消融
   1. Full MIMO-WM — 完整模型
   2. w/o GLU — 去掉GLU门控
   3. w/o Residual — 去掉残差连接
   4. w/o LayerNorm — 去掉LayerNorm

B. Backbone替换
   5. SSM→LSTM — 用LSTM替换SSM backbone
   6. SSM→GRU — 用GRU替换SSM backbone

C. 超参消融
   7. d_state=8/16/32/64 — SSM状态维度
   8. L=1/2/4 — 层数
   9. D=64/96/128/256 — 隐藏维度

数据集：Humanoid (348D)
种子：5个 [42, 123, 456, 789, 1024]
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

# ============================================================
# 数据加载
# ============================================================
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
# 模型定义
# ============================================================

class MIMOLayer(nn.Module):
    """MIMO层：SSM + GLU + LayerNorm + 残差"""
    def __init__(self, d_model, d_state=16, use_glu=True, use_residual=True, use_norm=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.use_glu = use_glu
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        if use_glu:
            self.gate = nn.Linear(d_model, d_model)
            self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        h = self.norm(x) if self.use_norm else x
        h = self.ssm(h)
        if self.use_glu:
            h = self.output(h) * torch.sigmoid(self.gate(h))
        return residual + h if self.use_residual else h


class MIMOWorldModel(nn.Module):
    """MIMO世界模型 — 完整版或消融版"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2,
                 use_glu=True, use_residual=True, use_norm=True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([
            MIMOLayer(d_model, d_state, use_glu=use_glu, use_residual=use_residual, use_norm=use_norm)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
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
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])


class LSTMBackboneModel(nn.Module):
    """用LSTM替换SSM backbone"""
    def __init__(self, state_dim, action_dim, d_model=128, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.backbone = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Sequential(
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
        h, _ = self.backbone(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])


class GRUBackboneModel(nn.Module):
    """用GRU替换SSM backbone"""
    def __init__(self, state_dim, action_dim, d_model=128, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.backbone = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Sequential(
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
        h, _ = self.backbone(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])


# ============================================================
# 训练+评估
# ============================================================
def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
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
        with torch.no_grad():
            vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
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

    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
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
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    RESULTS_FILE = 'experiments/ablation_mimo.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    # 消融配置
    configs = {
        # === A. 核心组件消融 ===
        'Full':         (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'w/o-GLU':      (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'use_glu': False, 'use_residual': True, 'use_norm': True}),
        'w/o-Residual': (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': False, 'use_norm': True}),
        'w/o-LayerNorm':(MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': False}),
        # === B. Backbone替换 ===
        'SSM→LSTM':     (LSTMBackboneModel,{'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'n_layers': 2}),
        'SSM→GRU':      (GRUBackboneModel, {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'n_layers': 2}),
        # === C. d_state消融 ===
        'N8':           (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 8,  'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'N16':          (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'N32':          (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 32, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'N64':          (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 64, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        # === D. 层数消融 ===
        'L1':           (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 1, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'L4':           (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 128, 'd_state': 16, 'n_layers': 4, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        # === E. 维度消融 ===
        'D64':          (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 64,  'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'D96':          (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 96,  'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
        'D256':         (MIMOWorldModel,   {'state_dim': 348, 'action_dim': 17, 'd_model': 256, 'd_state': 16, 'n_layers': 2, 'use_glu': True, 'use_residual': True, 'use_norm': True}),
    }

    # 旧key → 新key映射（兼容已有结果）
    OLD_KEY_MAP = {
        'Full': 'Full-MIMO-WM',
        'w/o-GLU': 'No-GLU',
        'L1': 'L1', 'L4': 'L4',
        'D64': 'D64', 'D96': 'D96', 'D256': 'D256',
    }

    print('\n' + '='*60, flush=True)
    print('MIMO-WM 消融实验 v2', flush=True)
    print('='*60, flush=True)

    for name, (ModelClass, kwargs) in configs.items():
        key = name
        # 检查新key或旧key是否有完整结果
        old_key = OLD_KEY_MAP.get(key)
        if key in results and len(results[key]) >= len(SEEDS):
            print(f'\n{name}: 已有完整结果，跳过', flush=True)
            continue
        if old_key and old_key in results and len(results[old_key]) >= len(SEEDS):
            results[key] = results[old_key]
            print(f'\n{name}: 从旧结果迁移，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        if key not in results:
            results[key] = {}

        for seed in SEEDS:
            seed_key = f'seed{seed}'
            if seed_key in results[key]:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue
            print(f'  seed={seed}...', end=' ', flush=True)
            r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[key][seed_key] = r
            print(f'MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params_m"]:.3f}M', flush=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 打印结果汇总
    print('\n' + '='*60, flush=True)
    print('消融实验结果汇总', flush=True)
    print('='*60, flush=True)
    print('{:<16} {:<18} {:<10} {:<10} {:<10}'.format('配置', 'MSE(×10⁻²)', 'R²', '参数(M)', '推理(ms)'))
    print('-'*70)

    for name in configs:
        key = name
        if key in results:
            valid = [results[key][s] for s in results[key] if 'mse' in results[key][s]]
            if valid:
                mses = [r['mse'] for r in valid]
                r2s = [r['r2'] for r in valid]
                params = valid[0]['params_m']
                inf_time = valid[0]['inf_time_ms']
                print('{:<16} {:.2f}±{:.2f}      {:.4f}    {:.3f}    {:.2f}'.format(
                    name, np.mean(mses)*100, np.std(mses)*100, np.mean(r2s), params, inf_time
                ))

    print('\nDone!', flush=True)
