"""Train all models with optimized settings: z-score normalization, T=32, 100 epochs."""
import torch, torch.nn as nn, numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel

device = torch.device('cuda')
STATE_DIM = 348
ACTION_DIM = 17

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    episodes = []
    for f in files:
        d2 = np.load(os.path.join(d, f))
        episodes.append((d2['states'], d2['actions']))
    return episodes

def make_step_data(episodes, T, mean=None, std=None):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        # Z-score normalize
        if mean is not None:
            st = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(st[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(st[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def compute_stats(episodes):
    """Compute mean and std across all episodes."""
    all_states = np.concatenate([st for st, _ in episodes], axis=0)
    return all_states.mean(axis=0), all_states.std(axis=0)

def train_model(ModelClass, name, kwargs, data_dir, T=32, epochs=100, lr=5e-4, bs=64):
    torch.manual_seed(42); np.random.seed(42)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    is_synthetic = 'synthetic' in data_dir
    max_train = 400 if is_synthetic else 930
    max_val = 100 if is_synthetic else 233
    
    eps_tr = load_episodes(data_dir, 'train', max_train)
    eps_vl = load_episodes(data_dir, 'val', max_val)
    
    # Compute normalization stats from training data
    mean, std = compute_stats(eps_tr)
    
    Xs, Xa, Y = make_step_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_step_data(eps_vl, T, mean, std)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{name}: {params:.2f}M params, train={len(Xs)}, val={len(Xv)}', flush=True)
    
    best_val = float('inf')
    patience = 20
    no_improve = 0
    
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        losses = []
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xs = torch.FloatTensor(Xs[bi]).to(device)
            xa = torch.FloatTensor(Xa[bi]).to(device)
            y = torch.FloatTensor(Y[bi]).to(device)
            pred = model(xs, xa)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sch.step()
        
        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(Xv).to(device)
            xav = torch.FloatTensor(Xav).to(device)
            yv = torch.FloatTensor(Yv).to(device)
            val_loss = loss_fn(model(xv, xav), yv).item()
        
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), f'experiments/{name}.pth')
        else:
            no_improve += 1
        
        if (ep+1) % 10 == 0 or ep == 0:
            print(f'  Epoch {ep+1}: train={np.mean(losses):.6f}, val={val_loss:.6f}, best={best_val:.6f}', flush=True)
        
        if no_improve >= patience:
            print(f'  Early stop at epoch {ep+1}', flush=True)
            break
    
    print(f'{name} done: best_val={best_val:.6f}', flush=True)
    return best_val, mean, std

def evaluate_model(ModelClass, name, kwargs, data_dir, mean, std, T=32, bs=64):
    model = ModelClass(**kwargs).to(device)
    model.load_state_dict(torch.load(f'experiments/{name}.pth'))
    model.eval()
    
    is_synthetic = 'synthetic' in data_dir
    max_val = 100 if is_synthetic else 233
    
    eps_vl = load_episodes(data_dir, 'val', max_val)
    Xv, Xav, Yv = make_step_data(eps_vl, T, mean, std)
    
    with torch.no_grad():
        xv = torch.FloatTensor(Xv).to(device)
        xav = torch.FloatTensor(Xav).to(device)
        yv = torch.FloatTensor(Yv).to(device)
        pred_v = model(xv, xav)
        mse = nn.MSELoss()(pred_v, yv).item()
        ss_res = torch.sum((yv - pred_v)**2).item()
        ss_tot = torch.sum((yv - torch.mean(yv, dim=0))**2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    with torch.no_grad():
        dx = torch.randn(bs, T, STATE_DIM).to(device)
        da = torch.randn(bs, T-1, ACTION_DIM).to(device)
        for _ in range(5): model(dx, da)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(20): model(dx, da)
        if device.type == 'cuda': torch.cuda.synchronize()
        infer_ms = (time.time() - start) / 20 * 1000
    
    return {'mse': mse, 'r2': r2, 'inference_ms': infer_ms, 'params': sum(p.numel() for p in model.parameters()) / 1e6}

datasets = {
    'synthetic': 'data/synthetic',
    'humanoid': 'data/humanoid',
}
models = {
    'LSTM-WM': (LSTMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'hidden_dim': 128, 'n_layers': 4}),
    'Trans-WM': (TransformerWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 64, 'nhead': 4, 'n_layers': 2}),
    'Mamba-WM': (MambaWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'd_state': 16, 'n_layers': 4}),
    'SSM-WM': (SSMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_state': 16, 'd_model': 128, 'n_layers': 4}),
}

all_results = {}
for ds_name, ds_path in datasets.items():
    print(f'\n{"="*60}\nTraining on {ds_name} (T=32, z-score, 100 epochs)\n{"="*60}', flush=True)
    ds_results = {}
    for mname, (MC, kwargs) in models.items():
        fname = f'{mname}_{ds_name}_v2'
        print(f'\nTraining {fname}...', flush=True)
        _, mean, std = train_model(MC, fname, kwargs, ds_path, T=32, epochs=100, lr=5e-4)
        print(f'Evaluating {fname}...', flush=True)
        r = evaluate_model(MC, fname, kwargs, ds_path, mean, std, T=32)
        ds_results[mname] = r
        print(f'  MSE={r["mse"]:.6f}, R²={r["r2"]:.4f}, Infer={r["inference_ms"]:.1f}ms, Params={r["params"]:.2f}M', flush=True)
    all_results[ds_name] = ds_results

with open('experiments/training_results_v2.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{"="*60}\nSUMMARY (optimized)\n{"="*60}', flush=True)
for ds, dr in all_results.items():
    print(f'\n{ds}:', flush=True)
    for mn, r in dr.items():
        print(f'  {mn:<12} MSE={r["mse"]:.6f} R²={r["r2"]:.4f} Infer={r["inference_ms"]:.1f}ms Params={r["params"]:.2f}M', flush=True)
