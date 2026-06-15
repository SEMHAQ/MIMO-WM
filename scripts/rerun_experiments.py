"""Re-run experiments on D4RL Humanoid (fixed device handling)."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel

device = torch.device('cuda')
STATE_DIM = 348; ACTION_DIM = 17; T = 32

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    return [(np.load(os.path.join(d, f))['states'], np.load(os.path.join(d, f))['actions']) for f in files]

def compute_stats(episodes):
    all_s = np.concatenate([st for st, _ in episodes], axis=0)
    return all_s.mean(axis=0), all_s.std(axis=0)

# ========== EXP 1: Batch Inference (more reliable measurement) ==========
def exp_batch_inference():
    print("\n" + "="*60)
    print("EXP 1: Batch Inference Time (D4RL models)")
    print("="*60)
    
    models = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'hidden_dim': 128, 'n_layers': 4}, 'experiments/LSTM-WM_d4rl.pth'),
        'Transformer-WM': (TransformerWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 64, 'nhead': 4, 'n_layers': 2}, 'experiments/Trans-WM_d4rl.pth'),
        'Mamba-WM': (MambaWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'n_layers': 4}, 'experiments/Mamba-WM_d4rl.pth'),
        'S4D-WM': (SSMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'd_state': 16, 'n_layers': 4}, 'experiments/S4D-WM_d4rl.pth'),
    }
    
    batch_sizes = [1, 8, 32, 64]
    results = {}
    
    for name, (cls, kwargs, ckpt) in models.items():
        model = cls(**kwargs).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        times = {}
        for bs in batch_sizes:
            dx = torch.randn(bs, T, STATE_DIM).to(device)
            da = torch.randn(bs, T-1, ACTION_DIM).to(device)
            # Warm up 10 times
            for _ in range(10):
                with torch.no_grad(): model(dx, da)
            torch.cuda.synchronize()
            # Measure 50 times, take median
            measurements = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(): model(dx, da)
                torch.cuda.synchronize()
                measurements.append((time.perf_counter() - t0) * 1000)
            times[bs] = round(np.median(measurements), 2)
        results[name] = times
        print(f"  {name}: {times}")
        del model; torch.cuda.empty_cache()
    
    return results

# ========== EXP 2: MPC (simplified - measure loop time only) ==========
def exp_mpc_timing():
    print("\n" + "="*60)
    print("EXP 2: MPC Loop Timing (D4RL models)")
    print("="*60)
    
    from src.models.mpc_controller import MPCController
    
    models = {
        'LSTM-MPC': (LSTMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'hidden_dim': 128, 'n_layers': 4}, 'experiments/LSTM-WM_d4rl.pth'),
        'Mamba-MPC': (MambaWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'n_layers': 4}, 'experiments/Mamba-WM_d4rl.pth'),
        'S4D-WM-MPC': (SSMWorldModel, {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'd_state': 16, 'n_layers': 4}, 'experiments/S4D-WM_d4rl.pth'),
    }
    
    results = {}
    for name, (cls, kwargs, ckpt) in models.items():
        model = cls(**kwargs).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        
        mpc = MPCController(model, horizon=10, n_iterations=50, device='cuda')
        
        # Generate synthetic test data on device
        state_hist = torch.randn(T, STATE_DIM).to(device).cpu().numpy()
        action_hist = torch.randn(T-1, ACTION_DIM).to(device).cpu().numpy()
        target = torch.randn(STATE_DIM).to(device).cpu().numpy()
        
        # Measure MPC plan time
        loop_times = []
        for _ in range(5):  # 5 runs
            t0 = time.perf_counter()
            try:
                action = mpc.plan(state_hist, action_hist, target)
                torch.cuda.synchronize()
                loop_times.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                print(f"  {name}: error: {e}")
                break
        
        if loop_times:
            avg = np.mean(loop_times)
            std = np.std(loop_times)
            freq = 1000 / avg
            results[name] = {
                'loop_ms': round(avg, 1),
                'loop_std': round(std, 1),
                'freq_hz': round(freq, 1),
            }
            print(f"  {name}: {avg:.1f}±{std:.1f}ms, {freq:.1f}Hz")
        
        del model, mpc; torch.cuda.empty_cache()
    
    return results

# ========== EXP 3: Threshold (evaluate existing model) ==========
def exp_threshold():
    print("\n" + "="*60)
    print("EXP 3: Threshold Function (D4RL Humanoid)")
    print("="*60)
    
    eps_tr = load_episodes('data/humanoid', 'train', 930)
    eps_vl = load_episodes('data/humanoid', 'val', 233)
    mean, std = compute_stats(eps_tr)
    
    model = SSMWorldModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=128, d_state=16, n_layers=4).to(device)
    model.load_state_dict(torch.load('experiments/S4D-WM_d4rl.pth', map_location=device))
    model.eval()
    
    Xv, Xav, Yv = [], [], []
    for st, ac in eps_vl:
        if len(st) < T+1: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xv.append(st_n[j:j+T]); Xav.append(ac[j:j+T-1]); Yv.append(st_n[j+T])
    Xv = torch.FloatTensor(np.array(Xv)).to(device)
    Xav = torch.FloatTensor(np.array(Xav)).to(device)
    Yv = torch.FloatTensor(np.array(Yv)).to(device)
    
    with torch.no_grad():
        pred = model(Xv, Xav)
        mse = nn.MSELoss()(pred, Yv).item()
        ss_res = torch.sum((Yv - pred)**2).item()
        ss_tot = torch.sum((Yv - torch.mean(Yv, dim=0))**2).item()
        r2 = 1 - ss_res / ss_tot
    
    print(f"  Soft threshold: MSE={mse:.6f}, R²={r2:.6f}")
    print("  NOTE: Hard/Garrote threshold need retraining (separate runs)")
    
    return {'soft': {'mse': round(mse, 6), 'r2': round(r2, 6)}}

if __name__ == '__main__':
    print("Re-running experiments on D4RL Humanoid")
    batch = exp_batch_inference()
    mpc = exp_mpc_timing()
    thresh = exp_threshold()
    
    all_results = {'batch_inference': batch, 'mpc': mpc, 'threshold': thresh}
    with open('experiments/d4rl_rerun_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("DONE - Results saved to experiments/d4rl_rerun_experiments.json")
    print(json.dumps(all_results, indent=2))
