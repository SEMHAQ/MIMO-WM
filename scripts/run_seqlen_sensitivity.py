"""序列长度敏感性分析: MIMO-WM在不同T下的预测精度"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import DiagSSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]

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
        for j in range(0, len(st)-T, max(T, 1)):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

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

class MIMOWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([MIMOLayer(d_model, d_state) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))
    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad = torch.zeros(states.shape[0], states.shape[1]-actions.shape[1], actions.shape[-1], device=actions.device)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for block in self.backbone:
            h = block(h)
        return states[:, -1, :] + self.decoder(h[:, -1, :])

def train_model(Xs, Xa, Y, Xv, Xav, Yv, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MIMOWorldModel(348, 17, d_model=96, d_state=16, n_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0
    for ep in range(100):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    return model

if __name__ == '__main__':
    TS = [8, 16, 32, 64, 128]
    RESULTS_FILE = 'experiments/seqlen_results.json'
    os.makedirs('experiments', exist_ok=True)
    results = {} if not os.path.exists(RESULTS_FILE) else json.load(open(RESULTS_FILE))

    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    mean, std = stats(eps_tr)

    for T in TS:
        key = f'MIMO-WM_T{T}'
        if key in results and len(results[key]) >= len(SEEDS):
            print(f'T={T}: 已有结果，跳过', flush=True)
            continue
        results[key] = {}
        print(f'\nT={T}', flush=True)
        Xs, Xa, Y = make_data(eps_tr, T, mean, std)
        Xv, Xav, Yv = make_data(eps_vl, T, mean, std)
        print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

        for seed in SEEDS:
            sk = f'seed{seed}'
            if sk in results[key]:
                print(f'  seed={seed}: 已有，跳过', flush=True)
                continue
            model = train_model(Xs, Xa, Y, Xv, Xav, Yv, seed=seed)
            model.eval()
            with torch.no_grad():
                Xv_g = torch.FloatTensor(Xv).to(device)
                Xav_g = torch.FloatTensor(Xav).to(device)
                Yv_g = torch.FloatTensor(Yv).to(device)
                pred = model(Xv_g, Xav_g)
                mse = nn.MSELoss()(pred, Yv_g).item()
                ss_r = ((Yv_g - pred)**2).sum().item()
                ss_t = ((Yv_g - Yv_g.mean(0))**2).sum().item()
                r2 = 1 - ss_r / ss_t
            results[key][sk] = {'mse': round(mse, 6), 'r2': round(r2, 4)}
            print(f'  seed={seed}: MSE={mse*100:.4f}, R2={r2:.4f}', flush=True)
            json.dump(results, open(RESULTS_FILE, 'w'), indent=2)

    print('\n' + '='*60)
    print('序列长度敏感性分析')
    print('='*60)
    for T in TS:
        key = f'MIMO-WM_T{T}'
        if key in results:
            vals = list(results[key].values())
            mses = [v['mse']*100 for v in vals]
            r2s = [v['r2'] for v in vals]
            print(f'T={T:<4} MSE={np.mean(mses):.4f}±{np.std(mses):.4f}  R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}')
    print('\nDone!', flush=True)
