"""Train all 5 models on Hopper-medium with 5 seeds.
Models: LSTM-WM, Transformer-WM, GRU-WM, Mamba-WM, S4D-WM
Seeds: 42, 123, 456, 789, 1024"""
import torch, torch.nn as nn
import numpy as np
import sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATE_DIM = 11; ACTION_DIM = 3; T = 32; BS = 64; EPOCHS = 100
SEEDS = [42, 123, 456, 789, 1024]

def load_episodes(data_dir, split):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def compute_stats(episodes):
    all_s = np.concatenate([st for st, _ in episodes], axis=0)
    return all_s.mean(axis=0), all_s.std(axis=0)

def make_data(episodes, T, mean, std, max_samples=None):
    Xs, Xa, Y = [], [], []
    for st, ac in episodes:
        if len(st) < T+1: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, max(1, T//2)):
            if j+T >= len(st): break
            Xs.append(st_n[j:j+T])
            # Pad or truncate actions to T-1
            a_seg = ac[j:j+T-1]
            if len(a_seg) < T-1:
                a_seg = np.pad(a_seg, ((0, T-1-len(a_seg)), (0,0)))
            Xa.append(a_seg)
            Y.append(st_n[j+T])
            if max_samples and len(Xs) >= max_samples: break
        if max_samples and len(Xs) >= max_samples: break
    return np.array(Xs), np.array(Xa), np.array(Y)

def train_one(model, Xs, Xa, Y, Xv, Xav, Yv, epochs, lr=5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    best_val_mse = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        train_loss = 0; n_batches = 0
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            s = torch.FloatTensor(Xs[bi]).to(device)
            a = torch.FloatTensor(Xa[bi]).to(device)
            y = torch.FloatTensor(Y[bi]).to(device)
            pred = model(s, a)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = []
            for i in range(0, len(Xv), BS):
                s = torch.FloatTensor(Xv[i:i+BS]).to(device)
                a = torch.FloatTensor(Xav[i:i+BS]).to(device)
                val_pred.append(model(s, a).cpu().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            val_mse = np.mean((val_pred - Yv) ** 2)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final eval: MSE and R²
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xv), BS):
            s = torch.FloatTensor(Xv[i:i+BS]).to(device)
            a = torch.FloatTensor(Xav[i:i+BS]).to(device)
            preds.append(model(s, a).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    mse = np.mean((preds - Yv) ** 2)
    ss_res = np.sum((Yv - preds) ** 2)
    ss_tot = np.sum((Yv - Yv.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return best_val_mse, r2

if __name__ == '__main__':
    print(f"=== Hopper Training: {STATE_DIM}D state, {ACTION_DIM}D action ===", flush=True)
    print(f"Device: {device}, Epochs: {EPOCHS}, Seeds: {SEEDS}", flush=True)
    
    eps_tr = load_episodes('data/hopper', 'train')
    eps_vl = load_episodes('data/hopper', 'val')
    mean, std = compute_stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, mean, std, max_samples=5000)
    Xv, Xav, Yv = make_data(eps_vl, T, mean, std, max_samples=2000)
    print(f"Train: {len(Xs)}, Val: {len(Xv)}", flush=True)
    
    results = {}
    
    model_configs = {
        'LSTM-WM': lambda seed: LSTMWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128, n_layers=4),
        'Transformer-WM': lambda seed: TransformerWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=64, nhead=4, n_layers=2),
        'GRU-WM': lambda seed: GRUWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128, n_layers=4),
        'Mamba-WM': lambda seed: MambaWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=128, d_state=16, n_layers=4),
        'S4D-WM': lambda seed: SSMWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=128, d_state=16, n_layers=4),
    }
    
    for model_name, model_fn in model_configs.items():
        for seed in SEEDS:
            key = f"{model_name}_hopper"
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = model_fn(seed).to(device)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            
            t0 = time.time()
            val_mse, r2 = train_one(model, Xs, Xa, Y, Xv, Xav, Yv, EPOCHS)
            elapsed = time.time() - t0
            
            # Save checkpoint
            ckpt_path = f'experiments/{model_name}_hopper_seed{seed}.pth'
            torch.save(model.state_dict(), ckpt_path)
            
            # Inference time
            model.eval()
            s_dummy = torch.FloatTensor(np.random.randn(1, T, STATE_DIM)).to(device)
            a_dummy = torch.FloatTensor(np.random.randn(1, T-1, ACTION_DIM)).to(device)
            # Warmup
            with torch.no_grad():
                for _ in range(10): model(s_dummy, a_dummy)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(100): model(s_dummy, a_dummy)
            torch.cuda.synchronize()
            inf_time = (time.time() - t0) / 100 * 1000
            
            if key not in results:
                results[key] = {}
            results[key][f'seed{seed}'] = {
                'mse': float(val_mse),
                'r2': float(r2),
                'params_m': float(n_params),
                'inf_ms': float(inf_time),
                'train_time_s': float(elapsed)
            }
            
            print(f"  {model_name} seed={seed}: MSE={val_mse:.6f}, R²={r2:.4f}, "
                  f"params={n_params:.2f}M, inf={inf_time:.1f}ms, time={elapsed:.0f}s", flush=True)
        
        # Save intermediate results
        with open('experiments/hopper_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Hopper Results Summary ===", flush=True)
    print(f"{'Model':<18} {'MSE':<12} {'R²':<10} {'Params':<10} {'Inf(ms)':<10}", flush=True)
    for model_name in model_configs:
        key = f"{model_name}_hopper"
        if key in results:
            mses = [v['mse'] for v in results[key].values()]
            r2s = [v['r2'] for v in results[key].values()]
            params = list(results[key].values())[0]['params_m']
            inf = list(results[key].values())[0]['inf_ms']
            print(f"{model_name:<18} {np.mean(mses):.4f}±{np.std(mses):.4f}  "
                  f"{np.mean(r2s):.4f}±{np.std(r2s):.4f}  {params:.2f}M  {inf:.1f}", flush=True)
    
    print("\nDone!", flush=True)
