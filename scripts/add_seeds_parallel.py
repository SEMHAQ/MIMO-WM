"""Add seeds 789 and 1024 - PARALLEL version (4 experiments simultaneously on GPU).
Uses threading to run multiple training jobs in parallel, filling GPU memory."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

device = torch.device('cuda')
NEW_SEEDS = [789, 1024]
EPOCHS = 100
BS = 64
LR = 5e-4
T = 32
PARALLEL = 4  # 同时跑4个实验

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

def train_one(args):
    """Train one model+seed combo. Returns (key, seed, result)."""
    ModelClass, name, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed = args
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

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
            vl = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1; torch.save(model.state_dict(), f'experiments/{name}_seed{seed}.pth')
        else: pat += 1
        if pat >= 20: break

    model.load_state_dict(torch.load(f'experiments/{name}_seed{seed}.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        mse = loss_fn(pred, torch.FloatTensor(Yv).to(device)).item()
        ss_r = torch.sum((torch.FloatTensor(Yv).to(device) - pred)**2).item()
        ss_t = torch.sum((torch.FloatTensor(Yv).to(device) - torch.mean(torch.FloatTensor(Yv).to(device), dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    # Free memory
    del model; torch.cuda.empty_cache()
    return name, seed, {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep}

# Load existing results
with open('experiments/multiseed_results.json') as f:
    results = json.load(f)

datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17, 'train_max': 930, 'val_max': 233},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8, 'train_max': 837, 'val_max': 210},
    'walker2d': {'dir': 'data/walker2d', 'sd': 17, 'ad': 6, 'train_max': 835, 'val_max': 209},
}

models = {
    'LSTM-WM': (LSTMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
    'Transformer-WM': (TransformerWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 64, 'nhead': 4, 'n_layers': 2}),
    'Mamba-WM': (MambaWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'n_layers': 4}),
    'S4D-WM': (SSMWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
    'GRU-WM': (GRUWorldModel, lambda sd, ad: {'state_dim': sd, 'action_dim': ad, 'hidden_dim': 128, 'n_layers': 4}),
}

# Build task list
tasks = []
for ds_name, ds_cfg in datasets.items():
    for model_name, (ModelClass, kwargs_fn) in models.items():
        key = f'{model_name}_{ds_name}'
        if key not in results:
            results[key] = {}
        missing = [s for s in NEW_SEEDS if f'seed{s}' not in results[key]]
        if not missing:
            continue
        kwargs = kwargs_fn(ds_cfg['sd'], ds_cfg['ad'])
        eps_tr = load_eps(ds_cfg['dir'], 'train', ds_cfg['train_max'])
        eps_vl = load_eps(ds_cfg['dir'], 'val', ds_cfg['val_max'])
        m, s = stats(eps_tr)
        Xs, Xa, Y = make_data(eps_tr, T, m, s)
        Xv, Xav, Yv = make_data(eps_vl, T, m, s)
        for seed in missing:
            tasks.append((ModelClass, key, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed))

print(f'Total tasks: {len(tasks)}, running {PARALLEL} in parallel')
print(f'GPU memory before: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated')

# Run in parallel
completed = 0
t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
    futures = {pool.submit(train_one, t): t for t in tasks}
    for future in as_completed(futures):
        key, seed, result = future.result()
        results[key][f'seed{seed}'] = result
        completed += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / completed * (len(tasks) - completed)
        print(f'  [{completed}/{len(tasks)}] {key} seed={seed}: MSE={result["mse"]:.6f} R²={result["r2"]:.6f} (ETA {eta/60:.0f}min)')
        # Save after each completion
        with open('experiments/multiseed_results.json', 'w') as f:
            json.dump(results, f, indent=2)

# Print summary
print('\n' + '='*60)
print('SUMMARY (5 seeds)')
print('='*60)
for key in sorted(results.keys()):
    seeds = results[key]
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    r2s = [seeds[s]['r2'] for s in sorted(seeds.keys())]
    n = len(mses)
    if n >= 2:
        print(f'{key} ({n}): MSE={np.mean(mses):.4f}±{np.std(mses, ddof=1):.4f}, R²={np.mean(r2s):.4f}±{np.std(r2s, ddof=1):.4f}')

total_time = time.perf_counter() - t_start
print(f'\nTotal time: {total_time/60:.1f}min')
print('Done!')
