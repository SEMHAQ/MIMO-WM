"""E4: Multi-step prediction error accumulation curves.
Evaluate S4D-WM and Mamba-WM at H=1,2,4,8,16 on Humanoid.
Plot MSE vs rollout step H."""
import torch, torch.nn as np_nn
import numpy as np
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATE_DIM = 348; ACTION_DIM = 17; T = 32; SEEDS = [42, 123, 456, 789, 1024]
H_VALUES = [1, 2, 4, 8, 16]

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def compute_stats(episodes):
    all_s = np.concatenate([st for st, _ in episodes], axis=0)
    return all_s.mean(axis=0), all_s.std(axis=0)

def make_ms_data(episodes, T, H, mean, std, max_samples=500):
    Xs, Xa, Ys = [], [], []
    for st, ac in episodes:
        if len(st) < T + H: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T-H+1, T):
            if j+T+H > len(st): break
            Xs.append(st_n[j:j+T])
            Xa.append(ac[j:j+T-1])
            Ys.append(st_n[j+T:j+T+H])
            if len(Xs) >= max_samples: break
        if len(Xs) >= max_samples: break
    return np.array(Xs), np.array(Xa), np.array(Ys)

def eval_multistep(model, Xs, Xa, Ys, H, device):
    """Evaluate multi-step prediction. Returns per-step MSE list."""
    model.eval()
    bs = 64
    all_mse = [[] for _ in range(H)]
    
    with torch.no_grad():
        for i in range(0, len(Xs), bs):
            s_batch = torch.FloatTensor(Xs[i:i+bs]).to(device)
            a_batch = torch.FloatTensor(Xa[i:i+bs]).to(device)
            y_batch = Ys[i:i+bs]  # (B, H, D)
            
            # Autoregressive rollout
            states_seq = s_batch.clone()
            actions_seq = a_batch.clone()
            
            for h in range(H):
                pred = model(states_seq, actions_seq)
                gt = torch.FloatTensor(y_batch[:, h, :]).to(device)
                mse_h = ((pred - gt) ** 2).mean(dim=1).cpu().numpy()
                all_mse[h].extend(mse_h)
                
                # Update for next step
                states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
                if h < H - 1:
                    # Need future action - use ground truth action from next step
                    # We don't have future actions in this setup, so we use zero padding
                    # Actually, the actions in the original data are for steps T-1 to T+H-2
                    # Let's handle this differently
                    pass
            
    return [np.mean(all_mse[h]) for h in range(H)]

def eval_multistep_v2(model, episodes, T, H, mean, std, device, max_samples=200):
    """Better multi-step eval: uses actual future actions from episodes."""
    model.eval()
    all_mse_h = []
    
    for st, ac in episodes:
        if len(st) < T + H + 1: continue
        st_n = (st - mean) / (std + 1e-8)
        
        for j in range(0, min(len(st)-T-H, max_samples*2), max(1, (len(st)-T-H)//max_samples)):
            if j+T+H >= len(st): break
            
            init_states = torch.FloatTensor(st_n[j:j+T]).unsqueeze(0).to(device)
            init_actions = torch.FloatTensor(ac[j:j+T-1]).unsqueeze(0).to(device)
            
            # Future actions for rollout
            future_actions = torch.FloatTensor(ac[j+T-1:j+T-1+H]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Autoregressive rollout
                states_seq = init_states.clone()
                actions_seq = init_actions.clone()
                step_mses = []
                
                for h in range(H):
                    pred = model(states_seq, actions_seq)
                    gt = torch.FloatTensor(st_n[j+T+h]).unsqueeze(0).to(device)
                    mse_h = ((pred - gt) ** 2).mean().item()
                    step_mses.append(mse_h)
                    
                    # Update sequences
                    states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
                    next_act = future_actions[:, h:h+1]
                    actions_seq = torch.cat([actions_seq[:, 1:], next_act], dim=1)
            
            all_mse_h.append(step_mses)
            if len(all_mse_h) >= max_samples:
                break
        if len(all_mse_h) >= max_samples:
            break
    
    all_mse_h = np.array(all_mse_h)
    return all_mse_h.mean(axis=0).tolist(), all_mse_h.std(axis=0).tolist()

if __name__ == '__main__':
    print("E4: Error accumulation curves on Humanoid", flush=True)
    
    # Load data
    eps_tr = load_episodes('data/humanoid', 'train', 930)
    eps_vl = load_episodes('data/humanoid', 'val', 233)
    mean, std = compute_stats(eps_tr)
    
    results = {}
    
    for model_name in ['S4D-WM', 'Mamba-WM']:
        results[model_name] = {}
        for seed in SEEDS:
            ckpt_path = f'experiments/{model_name}_humanoid_seed{seed}.pth'
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {ckpt_path} not found", flush=True)
                continue
            
            # Load model
            if model_name == 'S4D-WM':
                model = SSMWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, 
                                       d_model=128, d_state=16, n_layers=4).to(device)
            else:
                model = MambaWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                                         d_model=128, d_state=16, n_layers=4).to(device)
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            
            print(f"  {model_name} seed={seed}: evaluating H=1..16", flush=True)
            
            for H in H_VALUES:
                mse_mean, mse_std = eval_multistep_v2(model, eps_vl[:50], T, H, mean, std, device, max_samples=200)
                results[model_name].setdefault(f'H={H}', []).append({
                    'seed': seed,
                    'mse_per_step': mse_mean,
                    'mse_std_per_step': mse_std,
                    'mse_avg': np.mean(mse_mean)
                })
                print(f"    H={H}: avg_mse={np.mean(mse_mean):.4f}", flush=True)
    
    # Save results
    with open('experiments/error_accumulation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiments/error_accumulation.json", flush=True)
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Model':<12} {'H':<4} {'MSE_avg':<12} {'MSE_std':<12}")
    for model_name in ['S4D-WM', 'Mamba-WM']:
        for H in H_VALUES:
            key = f'H={H}'
            if key in results[model_name]:
                vals = [r['mse_avg'] for r in results[model_name][key]]
                print(f"{model_name:<12} {H:<4} {np.mean(vals):<12.4f} {np.std(vals):<12.4f}")
