"""实验3：不确定性估计（MIMO-WM独有）

MIMO-WM + 不确定性预测头
预测不确定性 → 用于安全控制
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
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

# ============================================================
# MIMO-WM + 不确定性估计
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

class MIMOWorldModelUncertainty(nn.Module):
    """MIMO-WM + 不确定性估计头"""
    def __init__(self, state_dim, action_dim, d_model=128, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim)
        )
        # 不确定性头：预测每个状态维度的方差
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, state_dim), nn.Softplus()  # 确保正值
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
        uncertainty = self.uncertainty_head(h_last)
        return state_pred, uncertainty

# ============================================================
# 训练+评估（NLL损失）
# ============================================================
def gaussian_nll(pred, target, var):
    """高斯负对数似然损失"""
    return 0.5 * torch.mean(torch.log(var + 1e-8) + (target - pred)**2 / (var + 1e-8))

def train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse_loss = nn.MSELoss()

    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0

    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            s_batch = torch.FloatTensor(Xs[bi]).to(device)
            a_batch = torch.FloatTensor(Xa[bi]).to(device)
            target = torch.FloatTensor(Y[bi]).to(device)

            state_pred, uncertainty = model(s_batch, a_batch)
            # 联合损失：MSE + NLL
            loss = mse_loss(state_pred, target) + 0.01 * gaussian_nll(state_pred, target, uncertainty)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            sp, _ = model(Xv_g, Xav_g)
            vl = mse_loss(sp, Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break

    # 评估
    model.eval()
    with torch.no_grad():
        sp, unc = model(Xv_g, Xav_g)
        state_mse = mse_loss(sp, Yv_g).item()
        ss_r = torch.sum((Yv_g - sp)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t

        # 校准度评估：不确定性是否准确反映实际误差
        actual_error = torch.mean((sp - Yv_g)**2, dim=-1)  # 实际MSE per sample
        predicted_var = torch.mean(unc, dim=-1)  # 预测方差 per sample
        # 校准度 = 预测方差与实际误差的相关系数
        corr = torch.corrcoef(torch.stack([actual_error, predicted_var]))[0, 1].item()

        # 覆盖率：在2σ范围内应该包含~95%的数据
        in_2sigma = torch.mean(((sp - Yv_g).abs() < 2 * torch.sqrt(unc + 1e-8)).float()).item()

    return {
        'state_mse': round(state_mse, 6), 'state_r2': round(r2, 4),
        'calibration_corr': round(corr, 4),
        'coverage_2sigma': round(in_2sigma, 4),
        'avg_uncertainty': round(torch.mean(unc).item(), 6),
        'params_m': round(params, 3), 'best_epoch': best_ep
    }

# ============================================================
# 主实验
# ============================================================
if __name__ == '__main__':
    datasets = {
        'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    }

    RESULTS_FILE = 'experiments/exp3_uncertainty.json'
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

        for seed in SEEDS:
            key = f'MIMO-Uncertainty_{ds_name}'
            if key not in results: results[key] = {}
            seed_key = f'seed{seed}'
            if seed_key in results[key]:
                print(f'  seed={seed} 已有，跳过', flush=True)
                continue
            print(f'  seed={seed}...', end=' ', flush=True)
            model = MIMOWorldModelUncertainty(ds_cfg['sd'], ds_cfg['ad'], d_model=96, d_state=16, n_layers=2)
            r = train_eval(model, Xs, Xa, Y, Xv, Xav, Yv, seed)
            results[key][seed_key] = r
            print(f'MSE={r["state_mse"]:.4f}, Corr={r["calibration_corr"]:.3f}, Coverage={r["coverage_2sigma"]:.3f}', flush=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # 汇总
    print('\n' + '='*60, flush=True)
    print('实验3：不确定性估计结果', flush=True)
    print('='*60, flush=True)

    for ds_name in datasets:
        key = f'MIMO-Uncertainty_{ds_name}'
        if key in results:
            valid = [results[key][s] for s in results[key] if 'state_mse' in results[key][s]]
            if valid:
                smse = [r['state_mse'] for r in valid]
                sr2 = [r['state_r2'] for r in valid]
                corr = [r['calibration_corr'] for r in valid]
                cov = [r['coverage_2sigma'] for r in valid]
                unc = [r['avg_uncertainty'] for r in valid]
                print(f'\n{ds_name}:', flush=True)
                print(f'  State MSE: {np.mean(smse)*100:.2f}±{np.std(smse)*100:.2f} (×10⁻²)', flush=True)
                print(f'  R²: {np.mean(sr2):.4f}±{np.std(sr2):.4f}', flush=True)
                print(f'  校准相关系数: {np.mean(corr):.4f}±{np.std(corr):.4f}', flush=True)
                print(f'  2σ覆盖率 (理想95%): {np.mean(cov)*100:.1f}%±{np.std(cov)*100:.1f}%', flush=True)
                print(f'  平均不确定性: {np.mean(unc):.6f}', flush=True)

    print('\nDone!', flush=True)
