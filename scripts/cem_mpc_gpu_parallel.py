"""CEM-MPC with GPU batch parallel: evaluate all K candidates in one forward pass.
This should be much faster than serial evaluation."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel

device = torch.device('cuda')
T = 32

datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17, 'train_max': 930, 'val_max': 233},
}

def load_model(model_name, dataset, seed=42):
    ds_cfg = datasets[dataset]
    if model_name == 'S4D-WM':
        model = SSMWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], d_model=128, d_state=16, n_layers=4)
    elif model_name == 'Mamba-WM':
        model = MambaWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], d_model=128, n_layers=4)
    elif model_name == 'LSTM-WM':
        model = LSTMWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], hidden_dim=128, n_layers=4)
    elif model_name == 'GRU-WM':
        model = GRUWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], hidden_dim=128, n_layers=4)
    path = f'experiments/{model_name}_{dataset}_seed{seed}.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def cem_mpc_gpu(model, init_states, init_actions, H=10, K=1000, n_elite=50, n_iter=5, action_std=0.5):
    """GPU-parallel CEM: all K candidates evaluated in one batch forward pass."""
    da = init_actions.shape[-1]
    ds = init_states.shape[-1]
    T_ctx = init_states.shape[1]

    # Expand init for K candidates: (K, T, D)
    states = init_states.expand(K, -1, -1).contiguous()
    actions = init_actions.expand(K, -1, -1).contiguous()

    mean = torch.zeros(H, da, device=device)
    std = torch.ones(H, da, device=device) * action_std

    t0 = time.perf_counter()
    for _ in range(n_iter):
        # Sample K candidates
        eps = torch.randn(K, H, da, device=device)
        action_seq = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        action_seq = action_seq.clamp(-1, 1)

        # Rollout all K candidates in one batch
        total_cost = torch.zeros(K, device=device)
        s = states.clone()
        a = actions.clone()
        for h in range(H):
            with torch.no_grad():
                pred = model(s, a)  # (K, ds)
            total_cost += torch.sum(pred ** 2, dim=-1)
            total_cost += 0.01 * torch.sum(action_seq[:, h] ** 2, dim=-1)
            s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
            a = torch.cat([a[:, 1:], action_seq[:, h:h+1]], dim=1)

        # Select elite
        elite_idx = total_cost.topk(n_elite, largest=False).indices
        elite = action_seq[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp(min=0.01)

    elapsed = time.perf_counter() - t0
    return mean, elapsed

def load_eps(d, s, mx=None):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    if mx: fs = fs[:mx]
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

def stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

# Load data
eps_tr = load_eps('data/humanoid', 'train', 930)
eps_vl = load_eps('data/humanoid', 'val', 233)
m, s = stats(eps_tr)
ep_states, ep_actions = eps_vl[0]
ep_states_n = (ep_states - m) / (s + 1e-8)

N_TRIALS = 5
H = 10
results = {}

for model_name in ['LSTM-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    print(f'\n{model_name}:')
    model = load_model(model_name, 'humanoid', seed=42)

    for K in [100, 500, 1000]:
        times = []
        for trial in range(N_TRIALS):
            t = min(trial * 50, len(ep_states_n) - T - 1)
            init_s = torch.FloatTensor(ep_states_n[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
            init_a = torch.FloatTensor(ep_actions[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
            _, elapsed = cem_mpc_gpu(model, init_s, init_a, H=H, K=K, n_elite=50, n_iter=5)
            times.append(elapsed * 1000)
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        hz = 1000 / mean_ms
        print(f'  K={K:4d}: {mean_ms:.1f}±{std_ms:.1f}ms ({hz:.1f}Hz)')
        results[f'{model_name}_K{K}'] = {'ms': round(mean_ms, 1), 'std': round(std_ms, 1), 'hz': round(hz, 1)}

with open('experiments/cem_mpc_gpu_parallel.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nDone!')
