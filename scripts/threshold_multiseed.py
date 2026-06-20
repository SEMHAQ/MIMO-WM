"""Threshold function comparison with 3 seeds (Table 11)."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel, DiagSSM

device = torch.device('cuda')
SEEDS = [42, 123, 456]
EPOCHS = 100
BS = 64; T = 32; LR = 5e-4

def load_eps(d, split, mx):
    eps = []
    fd = os.path.join(d, split)
    for f in sorted(os.listdir(fd))[:mx]:
        d2 = np.load(os.path.join(fd, f))
        eps.append((d2['states'], d2['actions']))
    return eps

eps_tr = load_eps('data/humanoid', 'train', 1000)
eps_vl = load_eps('data/humanoid', 'val', 163)
all_s = np.concatenate([e[0] for e in eps_tr])
mean, std = all_s.mean(0), all_s.std(0) + 1e-8

def make(eps, T, m, s):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        sn = (st - m) / s
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

Xs, Xa, Y = make(eps_tr, T, mean, std)
Xv, Xav, Yv = make(eps_vl, T, mean, std)
print(f"Train: {len(Xs)}, Val: {len(Xv)}")

# Define threshold variants by monkey-patching SSMBlock.forward
import src.models.ssm_world_model as ssm_mod

def make_forward(threshold_type):
    """Create a forward method with the specified threshold function."""
    def forward(self, x, mode='conv'):
        residual = x
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        raw_gate = self.gate(x_norm)
        if threshold_type == 'soft':
            g = torch.sigmoid(raw_gate)
        elif threshold_type == 'hard':
            g = (raw_gate > 0).float()
        elif threshold_type == 'garrote':
            # Garrote: g = 1 - (threshold / (x + eps))^2, clamped
            g = torch.sigmoid(raw_gate)  # use sigmoid as base, then apply garrote
            g = 1.0 - (0.5 / (g + 0.01))**2
            g = torch.clamp(g, 0, 1)
        out = g * ssm_out + (1 - g) * x_norm
        return residual + out
    return forward

def train_model(seed, threshold_type):
    torch.manual_seed(seed); np.random.seed(seed)
    # Monkey-patch
    original_forward = ssm_mod.SSMBlock.forward
    ssm_mod.SSMBlock.forward = make_forward(threshold_type)
    
    model = SSMWorldModel(state_dim=348, action_dim=17, d_model=128, d_state=16, n_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf'); pat = 0
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
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)),
                         torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    
    # Final eval
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    
    # Restore original
    ssm_mod.SSMBlock.forward = original_forward
    return {'mse': mse, 'r2': r2}

results = {}
for threshold in ['soft', 'hard', 'garrote']:
    results[threshold] = {}
    for seed in SEEDS:
        print(f"  {threshold} seed={seed}...", end=' ', flush=True)
        t0 = time.perf_counter()
        r = train_model(seed, threshold)
        elapsed = time.perf_counter() - t0
        results[threshold][f'seed{seed}'] = r
        print(f"MSE={r['mse']:.4f} R²={r['r2']:.4f} ({elapsed/60:.1f}min)")

# Save
with open('experiments/threshold_multiseed.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to experiments/threshold_multiseed.json")

# Print summary
import numpy as np
for t in ['soft', 'hard', 'garrote']:
    mses = [results[t][s]['mse'] for s in [f'seed{sd}' for sd in SEEDS]]
    r2s = [results[t][s]['r2'] for s in [f'seed{sd}' for sd in SEEDS]]
    print(f"{t}: MSE={np.mean(mses):.3f}±{np.std(mses):.3f}, R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")
