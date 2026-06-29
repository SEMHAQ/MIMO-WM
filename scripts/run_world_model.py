"""完整世界模型实验
创新点：
1. SSM-based世界模型
2. 多任务学习：状态+奖励+不确定性
3. 不确定性引导规划
4. 轻量级设计
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

# ============================================================
# 完整世界模型
# ============================================================
class WorldModel(nn.Module):
    """完整世界模型：状态预测 + 奖励预测 + 不确定性估计

    创新点：
    1. SSM-based动力学预测
    2. 多任务学习：状态+奖励+不确定性
    3. 轻量级设计
    """
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=1):
        super().__init__()
        self.state_dim = state_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # SSM动力学（单层，轻量）
        self.dynamics = nn.ModuleList([
            nn.ModuleDict({'norm': nn.LayerNorm(d_model), 'ssm': DiagSSM(d_model, d_state)})
            for _ in range(n_layers)
        ])

        # 状态预测头
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim)
        )

        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # 确保正值
        )

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1] - actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)

        # SSM动力学
        for block in self.dynamics:
            residual = h; x_norm = block['norm'](h); h = residual + block['ssm'](x_norm)

        h_last = h[:, -1, :]

        # 三个预测头
        state_pred = states[:, -1, :] + self.state_head(h_last)
        reward_pred = self.reward_head(h_last)
        uncertainty = self.uncertainty_head(h_last)

        return state_pred, reward_pred, uncertainty

# ============================================================
# 训练函数
# ============================================================
def train_world_model(model, Xs, Xa, Y, rewards, Xv, Xav, Yv, val_rewards, seed=SEEDS[0]):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()

    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    Rv_g = torch.FloatTensor(val_rewards).to(device)

    best_val = float('inf'); pat = 0; best_ep = 0

    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            s_batch = torch.FloatTensor(Xs[bi]).to(device)
            a_batch = torch.FloatTensor(Xa[bi]).to(device)
            target = torch.FloatTensor(Y[bi]).to(device)
            reward_target = torch.FloatTensor(rewards[bi]).to(device)

            state_pred, reward_pred, uncertainty = model(s_batch, a_batch)

            # 状态预测损失
            state_loss = state_loss_fn(state_pred, target)

            # 奖励预测损失
            reward_loss = reward_loss_fn(reward_pred.squeeze(), reward_target)

            # 不确定性正则化
            with torch.no_grad():
                state_error = torch.mean((state_pred - target)**2, dim=-1)
            uncertainty_loss = torch.mean((uncertainty.squeeze() - state_error)**2)

            # 总损失
            loss = state_loss + 0.1 * reward_loss + 0.01 * uncertainty_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            state_pred, _, _ = model(Xv_g, Xav_g)
            val_loss = state_loss_fn(state_pred, Yv_g).item()

        if val_loss < best_val: best_val = val_loss; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break

    # 评估
    model.eval()
    with torch.no_grad():
        state_pred, reward_pred, uncertainty = model(Xv_g, Xav_g)
        mse = state_loss_fn(state_pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - state_pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

    return {
        'state_mse': round(mse, 6),
        'state_r2': round(r2, 4),
        'params_m': round(params, 3),
        'best_epoch': best_ep
    }

# ============================================================
# 基线：简单SSM
# ============================================================
class SimpleSSM(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=1):
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

    # 生成奖励（简化：用状态变化作为奖励代理）
    rewards = np.mean(np.abs(Y - Xs[:, -1, :]), axis=-1)
    val_rewards = np.mean(np.abs(Yv - Xv[:, -1, :]), axis=-1)

    RESULTS_FILE = 'experiments/world_model_final.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    print('\n' + '='*60, flush=True)
    print('完整世界模型实验', flush=True)
    print('='*60, flush=True)

    # 测试不同配置
    configs = {
        'WM-d128-L1': lambda: WorldModel(348, 17, d_model=128, d_state=16, n_layers=1),
        'WM-d128-L2': lambda: WorldModel(348, 17, d_model=128, d_state=16, n_layers=2),
        'WM-d96-L1': lambda: WorldModel(348, 17, d_model=96, d_state=16, n_layers=1),
        'SimpleSSM': lambda: SimpleSSM(348, 17, d_model=128, d_state=16, n_layers=1),
    }

    for name, model_fn in configs.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        model = model_fn()
        r = train_world_model(model, Xs, Xa, Y, rewards, Xv, Xav, Yv, val_rewards)
        results[name] = r
        print(f'  State MSE={r["state_mse"]:.4f}, R²={r["state_r2"]:.4f}, Params={r["params_m"]:.3f}M', flush=True)

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
            print('{:<20} {:<10.4f} {:<10.4f} {:<10.3f}'.format(name, r['state_mse'], r['state_r2'], r['params_m']))

    print('\nDone!', flush=True)
