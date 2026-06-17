"""Sequence length search for Walker2d dataset."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
SEEDS = [42, 123, 456]
EPOCHS = 100
BS = 64
LR = 5e-4
SEQ_LENS = [16, 32, 64, 128, 256]

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

def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, T, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf'); pat = 0; best_ep = 0
    save_path = f'experiments/seqlen_walker2d_T{T}_seed{seed}.pth'
    
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
    
    # Eval
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    
    # Inference time
    sd, ad = kwargs['state_dim'], kwargs['action_dim']
    dummy_s = torch.randn(1, T, sd).to(device)
    dummy_a = torch.randn(1, T-1, ad).to(device)
    for _ in range(10):
        model(dummy_s, dummy_a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        model(dummy_s, dummy_a)
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) / 100 * 1000
    
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep,
            'infer_ms': round(infer_ms, 2)}

# Walker2d: 17D state, 6D action
print('Loading Walker2d data...')
eps_tr = load_eps('data/walker2d/train', None)
eps_vl = load_eps('data/walker2d/val', None)
m, s = stats(eps_tr)
sd, ad = 17, 6
print(f'Train eps: {len(eps_tr)}, Val eps: {len(eps_vl)}, State dim: {sd}')

results = {}
for T in SEQ_LENS:
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    if len(Xs) == 0 or len(Xv) == 0:
        print(f'  T={T}: insufficient data, skipping')
        continue
    print(f'\n  T={T}: Train={len(Xs)}, Val={len(Xv)}')
    
    results[str(T)] = {}
    for seed in SEEDS:
        kwargs = {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}
        print(f'    seed={seed}...', end=' ', flush=True)
        t0 = time.perf_counter()
        r = train_eval(SSMWorldModel, kwargs, Xs, Xa, Y, Xv, Xav, Yv, T, seed)
        elapsed = time.perf_counter() - t0
        results[str(T)][f'seed{seed}'] = r
        print(f'MSE={r["mse"]:.6f} R²={r["r2"]:.6f} ({elapsed/60:.1f}min)')
    
    with open('experiments/seqlen_walker2d.json', 'w') as f:
        json.dump(results, f, indent=2)

# Summary
print('\n' + '='*60)
print('SEQUENCE LENGTH SEARCH — Walker2d (S4D-WM, 3 seeds)')
print('='*60)
for T in SEQ_LENS:
    if str(T) not in results: continue
    mses = [results[str(T)][s]['mse'] for s in results[str(T)]]
    r2s = [results[str(T)][s]['r2'] for s in results[str(T)]]
    infer = [results[str(T)][s]['infer_ms'] for s in results[str(T)]]
    print(f'T={T:3d}: MSE={np.mean(mses):.4f}±{np.std(mses):.4f}, R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}, Infer={np.mean(infer):.1f}ms')

print('\nDone!')
