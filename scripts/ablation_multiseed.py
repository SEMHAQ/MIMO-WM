"""Multi-seed architecture ablation (Table 5): 3 seeds × 7 configs on Humanoid."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
SEEDS = [42, 123, 456]
EPOCHS = 100
BS = 64
LR = 5e-4
T = 32

def load_eps(d, s, mx=None):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    if mx: fs = fs[:mx]
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

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def train_ablation(config_name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    """Train one ablation config with one seed."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SSMWorldModel(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf'); pat = 0; best_ep = 0
    save_path = f'experiments/ablation_ms_{config_name}_seed{seed}.pth'
    
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
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val:
            best_val = vl; pat = 0; best_ep = ep+1
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
        if pat >= 20: break
    
    # Evaluate
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    
    # Inference time
    model.eval()
    dummy_s = torch.randn(1, T, 348).to(device)
    dummy_a = torch.randn(1, T-1, 17).to(device)
    # Warmup
    for _ in range(10):
        model(dummy_s, dummy_a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N_run = 100
    for _ in range(N_run):
        model(dummy_s, dummy_a)
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) / N_run * 1000
    
    params_m = count_params(model)
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep, 
            'infer_ms': round(infer_ms, 2), 'params_m': round(params_m, 4)}

# Humanoid dataset
print('Loading Humanoid data...')
eps_tr = load_eps('data/humanoid', 'train', 930)
eps_vl = load_eps('data/humanoid', 'val', 233)
m, s = stats(eps_tr)
Xs, Xa, Y = make_data(eps_tr, T, m, s)
Xv, Xav, Yv = make_data(eps_vl, T, m, s)
print(f'Train: {len(Xs)}, Val: {len(Xv)}')

# Ablation configs (matching paper Table 5)
sd, ad = 348, 17
configs = {
    'default':    {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4},
    'no_gate':    {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4},  # will modify model
    'no_res':     {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4},  # will modify model
    'L2':         {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 2},
    'L6':         {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 6},
    'N32':        {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 32, 'n_layers': 4},
    'D64':        {'state_dim': sd, 'action_dim': ad, 'd_model': 64,  'd_state': 16, 'n_layers': 4},
    'D256':       {'state_dim': sd, 'action_dim': ad, 'd_model': 256, 'd_state': 16, 'n_layers': 4},
}

# For no_gate and no_res, we need custom SSMBlock variants
# Monkey-patch approach: create modified model classes
import types

class SSMBlockNoGate(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        from src.models.ssm_world_model import DiagSSM
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
    def forward(self, x, mode='conv'):
        residual = x
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        return residual + ssm_out

class SSMBlockNoRes(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        from src.models.ssm_world_model import DiagSSM
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
    def forward(self, x, mode='conv'):
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        g = torch.sigmoid(self.gate(x_norm))
        return g * ssm_out + (1 - g) * x_norm

def make_model(config_name, kwargs):
    if config_name == 'no_gate':
        model = SSMWorldModel(**kwargs)
        model.backbone = nn.ModuleList([SSMBlockNoGate(kwargs['d_model'], kwargs['d_state']) for _ in range(kwargs['n_layers'])])
        return model.to(device)
    elif config_name == 'no_res':
        model = SSMWorldModel(**kwargs)
        model.backbone = nn.ModuleList([SSMBlockNoRes(kwargs['d_model'], kwargs['d_state']) for _ in range(kwargs['n_layers'])])
        return model.to(device)
    else:
        return SSMWorldModel(**kwargs).to(device)

# Modified train function for custom models
def train_ablation_custom(config_name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(config_name, kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf'); pat = 0; best_ep = 0
    save_path = f'experiments/ablation_ms_{config_name}_seed{seed}.pth'
    
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
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val:
            best_val = vl; pat = 0; best_ep = ep+1
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
        if pat >= 20: break
    
    # Evaluate
    model2 = make_model(config_name, kwargs)
    model2.load_state_dict(torch.load(save_path, map_location=device))
    model2.eval()
    with torch.no_grad():
        pred = model2(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    
    # Inference time
    model2.eval()
    dummy_s = torch.randn(1, T, 348).to(device)
    dummy_a = torch.randn(1, T-1, 17).to(device)
    for _ in range(10):
        model2(dummy_s, dummy_a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        model2(dummy_s, dummy_a)
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) / 100 * 1000
    
    params_m = sum(p.numel() for p in model2.parameters()) / 1e6
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep,
            'infer_ms': round(infer_ms, 2), 'params_m': round(params_m, 4)}

results = {}
for cfg_name, kwargs in configs.items():
    results[cfg_name] = {}
    for seed in SEEDS:
        print(f'  {cfg_name} seed={seed}...', end=' ', flush=True)
        t0 = time.perf_counter()
        if cfg_name in ('no_gate', 'no_res'):
            r = train_ablation_custom(cfg_name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
        else:
            r = train_ablation(cfg_name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed)
        elapsed = time.perf_counter() - t0
        results[cfg_name][f'seed{seed}'] = r
        print(f'MSE={r["mse"]:.6f} R²={r["r2"]:.6f} ({elapsed/60:.1f}min)')
    
    with open('experiments/ablation_multiseed.json', 'w') as f:
        json.dump(results, f, indent=2)

# Summary
print('\n' + '='*70)
print('ABLATION MULTI-SEED SUMMARY')
print('='*70)
for cfg_name in configs:
    mses = [results[cfg_name][s]['mse'] for s in [f'seed{sd}' for sd in SEEDS]]
    r2s = [results[cfg_name][s]['r2'] for s in [f'seed{sd}' for sd in SEEDS]]
    print(f'{cfg_name:12s}: MSE={np.mean(mses):.4f}±{np.std(mses):.4f}, R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}')

print('\nDone!')
