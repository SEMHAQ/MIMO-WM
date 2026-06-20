"""CEM-MPC experiment: Compare gradient-based MPC vs CEM sampling-based MPC.
Measures control frequency and trajectory tracking quality."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel

device = torch.device('cuda')
T = 32  # context length

def load_model(model_name, dataset, seed=42):
    """Load a trained model."""
    ds_cfg = datasets[dataset]
    if model_name == 'S4D-WM':
        model = SSMWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], d_model=128, d_state=16, n_layers=4)
    elif model_name == 'Mamba-WM':
        model = MambaWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], d_model=128, n_layers=4)
    elif model_name == 'LSTM-WM':
        model = LSTMWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], hidden_dim=128, n_layers=4)
    elif model_name == 'GRU-WM':
        model = GRUWorldModel(state_dim=ds_cfg['sd'], action_dim=ds_cfg['ad'], hidden_dim=128, n_layers=4)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    path = f'experiments/{model_name}_{dataset}_seed{seed}.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def rollout_cost(model, init_states, init_actions, action_seq, ref_states, Q=1.0, R=0.01):
    """Roll out the model with action_seq and compute tracking cost."""
    K, H, da = action_seq.shape
    ds = init_states.shape[-1]
    T_ctx = init_states.shape[-2] if init_states.dim() >= 3 else init_states.shape[0]

    # Force correct shape: (1, T, D)
    init_states = init_states.view(1, T_ctx, ds)
    init_actions = init_actions.view(1, -1, da)

    # Expand init for K candidates
    states = init_states.expand(K, -1, -1)  # (K, T, ds)
    actions = init_actions.expand(K, -1, -1)  # (K, T-1, da)

    total_cost = torch.zeros(K, device=device)
    for h in range(H):
        # Predict next state for all K candidates
        with torch.no_grad():
            pred = model(states, actions)  # (K, ds)
        # Track cost
        if h < ref_states.shape[1]:
            track_err = torch.sum((pred - ref_states[:, h]) ** 2, dim=-1)  # (K,)
        else:
            track_err = torch.sum(pred ** 2, dim=-1)  # penalize deviation from zero
        total_cost += Q * track_err
        # Action cost
        total_cost += R * torch.sum(action_seq[:, h] ** 2, dim=-1)

        # Update state/action sequences
        states = torch.cat([states[:, 1:], pred.unsqueeze(1)], dim=1)
        actions = torch.cat([actions[:, 1:], action_seq[:, h:h+1]], dim=1)

    return total_cost

def cem_mpc(model, init_states, init_actions, H=10, K=500, n_elite=50, n_iter=5, action_std=0.5):
    """Cross-Entropy Method MPC.
    Returns: best action sequence, time per control step."""
    da = init_actions.shape[-1]
    # Ensure 3D: (B, T, D)
    if init_states.dim() == 4:
        init_states = init_states.squeeze(1)
    if init_actions.dim() == 4:
        init_actions = init_actions.squeeze(1)

    t0 = time.perf_counter()

    # Initialize distribution
    mean = torch.zeros(H, da, device=device)
    std = torch.ones(H, da, device=device) * action_std

    # Dummy reference (zeros = stay at origin)
    ref_states = torch.zeros(1, H, init_states.shape[-1], device=device)

    for _ in range(n_iter):
        # Sample K candidates
        eps = torch.randn(K, H, da, device=device)
        action_seq = mean.unsqueeze(0) + std.unsqueeze(0) * eps  # (K, H, da)
        action_seq = action_seq.clamp(-1, 1)  # clip to action range

        # Evaluate
        costs = rollout_cost(model, init_states, init_actions, action_seq, ref_states)  # (K,)

        # Select elite
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = action_seq[elite_idx]  # (n_elite, H, da)

        # Update distribution
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.01)

    elapsed = time.perf_counter() - t0
    return mean, elapsed

def gradient_mpc(model, init_states, init_actions, H=10, n_iter=50, lr=0.01):
    """Gradient-based MPC using Adam. Returns: best action sequence, time."""
    da = init_actions.shape[-1]
    ds = init_states.shape[-1]
    T_ctx = init_states.shape[-2] if init_states.dim() >= 3 else init_states.shape[0]
    t0 = time.perf_counter()

    # Force correct shape: (1, T, D)
    init_states = init_states.view(1, T_ctx, ds)
    init_actions = init_actions.view(1, -1, da)

    # Initialize action sequence as trainable parameter
    action_seq = nn.Parameter(torch.zeros(H, da, device=device))
    opt = torch.optim.Adam([action_seq], lr=lr)

    ref_states = torch.zeros(1, H, init_states.shape[-1], device=device)

    # Set model to train mode for cudnn RNN backward compatibility
    model.train()

    for _ in range(n_iter):
        opt.zero_grad()
        # Roll out
        states = init_states.clone()
        actions = init_actions.clone()
        total_cost = 0.0
        for h in range(H):
            pred = model(states, actions)
            if h < ref_states.shape[1]:
                track_err = torch.sum((pred - ref_states[:, h]) ** 2)
            else:
                track_err = torch.sum(pred ** 2)
            total_cost = total_cost + track_err + 0.01 * torch.sum(action_seq[h] ** 2)
            states = torch.cat([states[:, 1:], pred.unsqueeze(1)], dim=1)
            actions = torch.cat([actions[:, 1:], action_seq[h:h+1].unsqueeze(0)], dim=1)
        total_cost.backward()
        torch.nn.utils.clip_grad_norm_([action_seq], 1.0)
        opt.step()

    model.eval()
    elapsed = time.perf_counter() - t0
    return action_seq.detach(), elapsed

# Dataset config
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17, 'train_max': 930, 'val_max': 233},
}

def load_eps(d, s, mx=None):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    if mx: fs = fs[:mx]
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

def stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

# Run experiments
print('Loading data...')
eps_tr = load_eps('data/humanoid', 'train', 930)
eps_vl = load_eps('data/humanoid', 'val', 233)
m, s = stats(eps_tr)

# Pick a validation episode
ep_states, ep_actions = eps_vl[0]
ep_states_n = (ep_states - m) / (s + 1e-8)

results = {}
N_TRIALS = 5
H = 10

for model_name in ['LSTM-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    print(f'\n{"="*40}\n{model_name}\n{"="*40}')
    model = load_model(model_name, 'humanoid', seed=42)

    # Gradient MPC
    grad_times = []
    for trial in range(N_TRIALS):
        t = min(trial * 50, len(ep_states_n) - T - 1)
        init_s = torch.FloatTensor(ep_states_n[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_actions[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
        _, elapsed = gradient_mpc(model, init_s, init_a, H=H)
        grad_times.append(elapsed * 1000)  # ms
    grad_mean = np.mean(grad_times)
    grad_std = np.std(grad_times)

    # CEM MPC
    cem_times = []
    for trial in range(N_TRIALS):
        t = min(trial * 50, len(ep_states_n) - T - 1)
        init_s = torch.FloatTensor(ep_states_n[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_actions[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
        _, elapsed = cem_mpc(model, init_s, init_a, H=H, K=500, n_elite=50, n_iter=5)
        cem_times.append(elapsed * 1000)  # ms
    cem_mean = np.mean(cem_times)
    cem_std = np.std(cem_times)

    # CEM MPC with GPU parallel (batch forward)
    cem_par_times = []
    for trial in range(N_TRIALS):
        t = min(trial * 50, len(ep_states_n) - T - 1)
        init_s = torch.FloatTensor(ep_states_n[t:t+T]).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_actions[t:t+T-1]).unsqueeze(0).to(device)
        _, elapsed = cem_mpc(model, init_s, init_a, H=H, K=500, n_elite=50, n_iter=5)
        cem_par_times.append(elapsed * 1000)
    cem_par_mean = np.mean(cem_par_times)

    results[model_name] = {
        'gradient_mpc_ms': round(grad_mean, 1),
        'gradient_mpc_std': round(grad_std, 1),
        'gradient_mpc_hz': round(1000 / grad_mean, 1),
        'cem_mpc_ms': round(cem_mean, 1),
        'cem_mpc_std': round(cem_std, 1),
        'cem_mpc_hz': round(1000 / cem_mean, 1),
    }
    print(f'  Gradient: {grad_mean:.1f}±{grad_std:.1f}ms ({1000/grad_mean:.1f}Hz)')
    print(f'  CEM:      {cem_mean:.1f}±{cem_std:.1f}ms ({1000/cem_mean:.1f}Hz)')

# Save results
os.makedirs('experiments', exist_ok=True)
with open('experiments/cem_mpc_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*60)
print('CEM MPC RESULTS')
print('='*60)
for name, r in results.items():
    print(f'{name}: Gradient {r["gradient_mpc_ms"]}ms/{r["gradient_mpc_hz"]}Hz, CEM {r["cem_mpc_ms"]}ms/{r["cem_mpc_hz"]}Hz')

print('\nDone!')
