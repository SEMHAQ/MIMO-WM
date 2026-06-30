"""实验2：奖励预测（MIMO-WM独有）

MIMO-WM + 奖励预测头
展示完整世界模型能力：状态 + 奖励联合预测
数据集：Humanoid, Ant, Walker2d
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
    Xs, Xa, Y, R = [], [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T])
            Xa.append(ac[j:j+T-1])
            Y.append(sn[j+T])
            # 奖励 = 状态变化量的L2范数（模拟环境奖励）
            R.append(np.linalg.norm(sn[j+T] - sn[j+T-1]))
    return np.array(Xs), np.array(Xa), np.array(Y), np.array(R)

# ============================================================
# MIMO-WM + 奖励预测
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

class MIMOWorldModelWithReward(nn.Module):
    """MIMO-WM + 奖励预测头"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        # 状态预测头
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        h_last = h[:, -1, :]
        state_pred = states[:, -1, :] + self.state_head(h_last)
        reward_pred = self.reward_head(h_last).squeeze(-1)
        return state_pred, reward_pred

# ============================================================
# 基线：无奖励预测的MIMO-WM
# ============================================================
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
# 训练+评估
# ============================================================
def train_eval_reward(model, Xs, Xa, Y, R, Xv, Xav, Yv, Rv, seed, reward_weight=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()

    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    Rv_g = torch.FloatTensor(Rv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0

    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            s_batch = torch.FloatTensor(Xs[bi]).to(device)
            a_batch = torch.FloatTensor(Xa[bi]).to(device)
            target = torch.FloatTensor(Y[bi]).to(device)
            r_target = torch.FloatTensor(R[bi]).to(device)

            state_pred, reward_pred = model(s_batch, a_batch)
            loss = state_loss_fn(state_pred, target) + reward_weight * reward_loss_fn(reward_pred, r_target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            sp, rp = model(Xv_g, Xav_g)
            vl = state_loss_fn(sp, Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break

    model.eval()
    with torch.no_grad():
        sp, rp = model(Xv_g, Xav_g)
        state_mse = state_loss_fn(sp, Yv_g).item()
        reward_mse = reward_loss_fn(rp, Rv_g).item()
        ss_r = torch.sum((Yv_g - sp)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
        # 奖励R²
        ss_rr = torch.sum((Rv_g - rp)**2).item()
        ss_rt = torch.sum((Rv_g - torch.mean(Rv_g))**2).item()
        reward_r2 = 1 - ss_rr / (ss_rt + 1e-8)

    return {
        'state_mse': round(state_mse, 6), 'state_r2': round(r2, 4),
        'reward_mse': round(reward_mse, 6), 'reward_r2': round(reward_r2, 4),
        'params_m': round(params, 3), 'best_epoch': best_ep
    }

def train_eval_baseline(model, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
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

    return {'state_mse': round(mse, 6), 'state_r2': round(r2, 4), 'params_m': round(params, 3), 'best_epoch': best_ep}

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    }

    RESULTS_FILE = 'experiments/exp2_reward.json'
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
        Xs, Xa, Y, R = make_data(eps_tr, T, m, s)
        Xv, Xav, Yv, Rv = make_data(eps_vl, T, m, s)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        sd, ad = ds_cfg['sd'], ds_cfg['ad']

        # 基线：无奖励预测
        for seed in SEEDS:
            key = f'MIMO-NoReward_{ds_name}'
            if key not in results: results[key] = {}
            seed_key = f'seed{seed}'
            if seed_key in results[key]:
                print(f'  MIMO-NoReward seed={seed} 已有，跳过', flush=True)
                continue
            print(f'  MIMO-NoReward seed={seed}...', end=' ', flush=True)
            model = MIMOWorldModel(sd, ad, d_model=96, d_state=16, n_layers=2)
            r = train_eval_baseline(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[key][seed_key] = r
            print(f'State MSE={r["state_mse"]:.4f}', flush=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

        # MIMO-WM + 奖励预测（不同权重）
        REWARD_WEIGHTS = [0.001, 0.01, 0.05, 0.1]
        for w in REWARD_WEIGHTS:
            for seed in SEEDS:
                key = f'MIMO-Reward-w{w}_{ds_name}'
                if key not in results: results[key] = {}
                seed_key = f'seed{seed}'
                if seed_key in results[key]:
                    print(f'  MIMO-Reward(w={w}) seed={seed} 已有，跳过', flush=True)
                    continue
                print(f'  MIMO-Reward(w={w}) seed={seed}...', end=' ', flush=True)
                model = MIMOWorldModelWithReward(sd, ad, d_model=96, d_state=16, n_layers=2)
                r = train_eval_reward(model, Xs, Xa, Y, R, Xv, Xav, Yv, Rv, seed, reward_weight=w)
                results[key][seed_key] = r
                print(f'State MSE={r["state_mse"]:.4f}, Reward R²={r["reward_r2"]:.4f}', flush=True)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    # 汇总
    print('\n' + '='*60, flush=True)
    print('实验2：奖励预测结果', flush=True)
    print('='*60, flush=True)

    for ds_name in datasets:
        print(f'\n{ds_name}:', flush=True)
        print('{:<22} {:<18} {:<12} {:<12}'.format('配置', 'State MSE(×10⁻²)', 'State R²', 'Reward R²'))
        print('-'*68)

        # 基线
        key = f'MIMO-NoReward_{ds_name}'
        if key in results:
            valid = [results[key][s] for s in results[key] if 'state_mse' in results[key][s]]
            if valid:
                smse = [r['state_mse'] for r in valid]
                sr2 = [r['state_r2'] for r in valid]
                print('{:<22} {:.2f}±{:.2f}        {:.4f}    {}'.format(
                    'No-Reward', np.mean(smse)*100, np.std(smse)*100, np.mean(sr2), '-'))

        # 不同权重
        for w in [0.001, 0.01, 0.05, 0.1]:
            key = f'MIMO-Reward-w{w}_{ds_name}'
            if key in results:
                valid = [results[key][s] for s in results[key] if 'state_mse' in results[key][s]]
                if valid:
                    smse = [r['state_mse'] for r in valid]
                    sr2 = [r['state_r2'] for r in valid]
                    rr2 = [r['reward_r2'] for r in valid]
                    print('{:<22} {:.2f}±{:.2f}        {:.4f}    {:.4f}'.format(
                        f'Reward-w={w}', np.mean(smse)*100, np.std(smse)*100, np.mean(sr2), np.mean(rr2)))

    print('\nDone!', flush=True)
