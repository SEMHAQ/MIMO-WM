"""Train S4D-WM for each T value on both D4RL datasets."""
import torch, torch.nn as nn, numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def make_step_data(episodes, T, mean=None, std=None):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        if mean is not None: st = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def compute_stats(episodes):
    all_states = np.concatenate([st for st, _ in episodes], axis=0)
    return all_states.mean(axis=0), all_states.std(axis=0)

def train_and_evaluate(dataset_name, state_dim, action_dim, T, epochs=100, lr=5e-4, bs=64):
    """Train S4D-WM for a specific T value and evaluate."""
    data_dir = f'data/{dataset_name}'
    torch.manual_seed(42); np.random.seed(42)
    
    max_train = 400 if dataset_name == 'humanoid' else 837
    max_val = 100 if dataset_name == 'humanoid' else 210
    
    eps_tr = load_episodes(data_dir, 'train', max_train)
    eps_vl = load_episodes(data_dir, 'val', max_val)
    mean, std = compute_stats(eps_tr)
    
    Xs, Xa, Y = make_step_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_step_data(eps_vl, T, mean, std)
    
    if len(Xs) == 0 or len(Xv) == 0:
        return None
    
    model = SSMWorldModel(state_dim=state_dim, action_dim=action_dim, d_state=16, d_model=128, n_layers=4).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_val = float('inf')
    patience = 20
    no_improve = 0
    save_path = f'experiments/S4D-WM_{dataset_name}_T{T}.pth'
    
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        losses = []
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xs = torch.FloatTensor(Xs[bi]).to(device)
            xa = torch.FloatTensor(Xa[bi]).to(device)
            y = torch.FloatTensor(Y[bi]).to(device)
            loss = loss_fn(model(xs, xa), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sch.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()
        
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    # Evaluate
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    with torch.no_grad():
        pred_v = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        yv = torch.FloatTensor(Yv).to(device)
        mse = nn.MSELoss()(pred_v, yv).item()
        ss_res = torch.sum((yv - pred_v)**2).item()
        ss_tot = torch.sum((yv - torch.mean(yv, dim=0))**2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Inference time
    with torch.no_grad():
        dx = torch.randn(64, T, state_dim).to(device)
        da = torch.randn(64, T-1, action_dim).to(device)
        for _ in range(5): model(dx, da)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20): model(dx, da)
        torch.cuda.synchronize()
        infer_ms = (time.time() - start) / 20 * 1000
    
    result = {
        'T': T, 'mse': mse, 'r2': r2, 'infer_ms': infer_ms, 
        'params': params, 'n_train': len(Xs), 'n_val': len(Xv)
    }
    
    print(f'{dataset_name} T={T}: MSE={mse:.6f}, R²={r2:.4f}, Infer={infer_ms:.1f}ms', flush=True)
    return result

# Training configurations
configs = [
    {'dataset': 'humanoid', 'state_dim': 348, 'action_dim': 17},
    {'dataset': 'ant', 'state_dim': 105, 'action_dim': 8},
]
T_values = [16, 32, 64, 128, 256]

all_results = {}
for cfg in configs:
    dataset = cfg['dataset']
    all_results[dataset] = []
    
    print(f'\n{"="*60}\nTraining on {dataset}\n{"="*60}', flush=True)
    
    for T in T_values:
        result = train_and_evaluate(
            dataset, cfg['state_dim'], cfg['action_dim'], T,
            epochs=100, lr=5e-4, bs=64
        )
        if result:
            all_results[dataset].append(result)

# Save results
with open('experiments/seqlen_results_final.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary
print(f'\n{"="*60}\nSUMMARY\n{"="*60}', flush=True)
for dataset, results in all_results.items():
    print(f'\n{dataset}:', flush=True)
    print(f'{"T":<6} {"MSE":<12} {"R²":<8} {"Infer(ms)":<12}', flush=True)
    print('-' * 40, flush=True)
    for r in results:
        print(f'{r["T"]:<6} {r["mse"]:<12.6f} {r["r2"]:<8.4f} {r["infer_ms"]:<12.1f}', flush=True)
    
    best = min(results, key=lambda x: x['mse'])
    print(f'\nBest: T={best["T"]} with MSE={best["mse"]:.6f}, R²={best["r2"]:.4f}', flush=True)
