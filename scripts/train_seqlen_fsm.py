"""Train FSM-WM for each T value on Ant dataset (based on train_seqlen_final.py)."""
import torch, torch.nn as nn, numpy as np, sys, os, time, json
sys.path.insert(0, '.')
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_and_evaluate(dataset_name, state_dim, action_dim, T, epochs=100, lr=5e-4, bs=256, seed=42):
    """Train FSM-WM for a specific T value and evaluate."""
    data_dir = f'data/{dataset_name}'
    torch.manual_seed(seed); np.random.seed(seed)

    eps_tr = load_episodes(data_dir, 'train')
    eps_vl = load_episodes(data_dir, 'val')
    mean, std = compute_stats(eps_tr)

    Xs, Xa, Y = make_step_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_step_data(eps_vl, T, mean, std)

    if len(Xs) == 0 or len(Xv) == 0:
        return None

    # FSM-WM配置
    model = FSM(state_dim=state_dim, action_dim=action_dim, d_model=128, d_state=16, n_layers=2, window_size=8).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    patience = 20
    no_improve = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xs = torch.FloatTensor(Xs[bi]).to(device)
            xa = torch.FloatTensor(Xa[bi]).to(device)
            y = torch.FloatTensor(Y[bi]).to(device)
            loss = loss_fn(model(xs, xa), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device)), torch.FloatTensor(Yv).to(device)).item()

        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Evaluate
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_v = model(torch.FloatTensor(Xv).to(device), torch.FloatTensor(Xav).to(device))
        yv = torch.FloatTensor(Yv).to(device)
        mse = nn.MSELoss()(pred_v, yv).item()
        ss_res = torch.sum((yv - pred_v)**2).item()
        ss_tot = torch.sum((yv - torch.mean(yv, dim=0))**2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    result = {
        'T': T, 'mse': mse, 'r2': r2,
        'params': params, 'n_train': len(Xs), 'n_val': len(Xv)
    }

    print(f'{dataset_name} T={T}: MSE={mse:.6f}, R²={r2:.4f}', flush=True)
    return result

# 只跑Ant
dataset = 'ant'
state_dim = 105
action_dim = 8
T_values = [16, 32, 64, 128, 256]

print(f'\n{"="*60}\nTraining FSM-WM on {dataset}\n{"="*60}', flush=True)

results = []
for T in T_values:
    result = train_and_evaluate(dataset, state_dim, action_dim, T, epochs=100, lr=5e-4, bs=256, seed=42)
    if result:
        results.append(result)

# Save results
with open('experiments/ant_seqlen_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f'\n{"="*60}\nSUMMARY\n{"="*60}', flush=True)
print(f'{"T":<6} {"MSE":<12} {"R²":<8}', flush=True)
print('-' * 30, flush=True)
for r in results:
    print(f'{r["T"]:<6} {r["mse"]:<12.6f} {r["r2"]:<8.4f}', flush=True)

best = min(results, key=lambda x: x['mse'])
print(f'\nBest: T={best["T"]} with MSE={best["mse"]:.6f}, R²={best["r2"]:.4f}', flush=True)
