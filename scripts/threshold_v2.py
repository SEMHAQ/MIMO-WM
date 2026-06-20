"""Threshold 3-seed: soft/hard/garrote. Simple direct approach."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')

device = torch.device('cuda')
SEEDS = [42, 123, 456]
EPOCHS = 100; BS = 64; T = 32; LR = 5e-4

import sys
log = open('/tmp/threshold_log.txt', 'w', buffering=1)  # line-buffered
def p(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()
p('Loading data...')
def load_eps(d, split, mx):
    eps = []
    for f in sorted(os.listdir(os.path.join(d, split)))[:mx]:
        d2 = np.load(os.path.join(d, split, f))
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
p(f'Train: {len(Xs)}, Val: {len(Xv)}')

# Build model with configurable threshold
from src.models.ssm_world_model import DiagSSM

class ThresholdSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, threshold_type):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.threshold_type = threshold_type
    def forward(self, x, mode='conv'):
        residual = x; x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        raw = self.gate(x_norm)
        if self.threshold_type == 'soft':
            g = torch.sigmoid(raw)
        elif self.threshold_type == 'hard':
            g = (raw > 0).float()
        else:  # garrote
            g = torch.sigmoid(raw)
            g = 1.0 - (0.5 / (g + 0.01))**2
            g = torch.clamp(g, 0, 1)
        return residual + g * ssm_out + (1 - g) * x_norm

class ThresholdWM(nn.Module):
    def __init__(self, state_dim, action_dim, d_model, d_state, n_layers, threshold_type):
        super().__init__()
        self.encoder = nn.Linear(state_dim + action_dim, d_model)
        self.blocks = nn.ModuleList([ThresholdSSMBlock(d_model, d_state, threshold_type) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, state_dim)
        self.residual = nn.Linear(state_dim, state_dim)
    def forward(self, states, actions, mode='conv'):
        # Pad actions to match states length
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1],
                              device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for blk in self.blocks:
            x = blk(x, mode=mode)
        return self.residual(states[:, -1]) + self.decoder(x[:, -1])

def train_one(seed, threshold_type):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ThresholdWM(348, 17, 128, 16, 4, threshold_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_val = float('inf'); pat = 0
    for ep in range(EPOCHS):
        model.train(); idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)),
                         torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    return {'mse': mse, 'r2': r2}

results = {}
for threshold in ['soft', 'hard', 'garrote']:
    results[threshold] = {}
    for seed in SEEDS:
        p(f'{threshold} seed={seed}...')
        t0 = time.perf_counter()
        r = train_one(seed, threshold)
        elapsed = time.perf_counter() - t0
        results[threshold][f'seed{seed}'] = r
        p(f'  MSE={r["mse"]:.4f} R2={r["r2"]:.4f} ({elapsed/60:.1f}min)')

with open('experiments/threshold_multiseed.json', 'w') as f:
    json.dump(results, f, indent=2)
p('Saved!')
for t in ['soft', 'hard', 'garrote']:
    mses = [results[t][s]['mse'] for s in results[t]]
    r2s = [results[t][s]['r2'] for s in results[t]]
    p(f'{t}: MSE={np.mean(mses):.3f}+-{np.std(mses):.3f}, R2={np.mean(r2s):.3f}+-{np.std(r2s):.3f}')
