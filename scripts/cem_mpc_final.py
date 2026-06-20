"""CEM-MPC final experiment: GPU batch parallel with optimized config.
K=200, H=10, 3 iterations → near 10Hz for S4D-WM."""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel

device = torch.device('cuda')
T = 32
H = 10
K = 200
N_ELITE = 30
N_ITER = 3

datasets = {'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17}}

def load_model(model_name, dataset, seed=42):
    ds = datasets[dataset]
    if model_name == 'S4D-WM':
        model = SSMWorldModel(state_dim=ds['sd'], action_dim=ds['ad'], d_model=128, d_state=16, n_layers=4)
    elif model_name == 'Mamba-WM':
        model = MambaWorldModel(state_dim=ds['sd'], action_dim=ds['ad'], d_model=128, n_layers=4)
    elif model_name == 'LSTM-WM':
        model = LSTMWorldModel(state_dim=ds['sd'], action_dim=ds['ad'], hidden_dim=128, n_layers=4)
    elif model_name == 'GRU-WM':
        model = GRUWorldModel(state_dim=ds['sd'], action_dim=ds['ad'], hidden_dim=128, n_layers=4)
    model.load_state_dict(torch.load(f'experiments/{model_name}_{dataset}_seed{seed}.pth', map_location=device))
    model.to(device).eval()
    return model

def cem_mpc(model, init_states, init_actions):
    da = init_actions.shape[-1]
    states = init_states.expand(K, -1, -1).contiguous()
    actions = init_actions.expand(K, -1, -1).contiguous()
    mean = torch.zeros(H, da, device=device)
    std = torch.ones(H, da, device=device) * 0.5

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        eps = torch.randn(K, H, da, device=device)
        act = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(-1, 1)
        cost = torch.zeros(K, device=device)
        s, a = states.clone(), actions.clone()
        for h in range(H):
            with torch.no_grad(): pred = model(s, a)
            cost += torch.sum(pred**2, dim=-1) + 0.01 * torch.sum(act[:, h]**2, dim=-1)
            s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
            a = torch.cat([a[:, 1:], act[:, h:h+1]], dim=1)
        idx = cost.topk(N_ELITE, largest=False).indices
        el = act[idx]; mean = el.mean(0); std = el.std(0).clamp(min=0.01)
    return (time.perf_counter() - t0) * 1000

def gradient_mpc(model, init_states, init_actions):
    da = init_actions.shape[-1]
    action_seq = nn.Parameter(torch.zeros(H, da, device=device))
    opt = torch.optim.Adam([action_seq], lr=0.01)
    model.train()
    t0 = time.perf_counter()
    for _ in range(50):
        opt.zero_grad()
        s, a = init_states.clone(), init_actions.clone()
        cost = 0.0
        for h in range(H):
            pred = model(s, a)
            cost = cost + torch.sum(pred**2) + 0.01 * torch.sum(action_seq[h]**2)
            s = torch.cat([s[:, 1:], pred.unsqueeze(1)], dim=1)
            a = torch.cat([a[:, 1:], action_seq[h:h+1].unsqueeze(0)], dim=1)
        cost.backward()
        torch.nn.utils.clip_grad_norm_([action_seq], 1.0); opt.step()
    model.eval()
    return (time.perf_counter() - t0) * 1000

# Load data
eps_vl = [(np.load(os.path.join('data/humanoid/val', f))['states'], np.load(os.path.join('data/humanoid/val', f))['actions']) for f in sorted(os.listdir('data/humanoid/val'))[:5]]
eps_tr = [(np.load(os.path.join('data/humanoid/train', f))['states'], np.load(os.path.join('data/humanoid/train', f))['actions']) for f in sorted(os.listdir('data/humanoid/train'))[:10]]
m = np.concatenate([s for s,_ in eps_tr]).mean(0)
std = np.concatenate([s for s,_ in eps_tr]).std(0)

N_TRIALS = 5
results = {}

for model_name in ['LSTM-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    print(f'\n{model_name}:')
    model = load_model(model_name, 'humanoid', seed=42)

    # Gradient MPC
    grad_times = []
    for trial in range(N_TRIALS):
        ep_s, ep_a = eps_vl[trial % len(eps_vl)]
        ep_sn = (ep_s - m) / (std + 1e-8)
        t = min(trial * 50, len(ep_sn) - T - 1)
        init_s = torch.FloatTensor(ep_sn[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_a[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
        grad_times.append(gradient_mpc(model, init_s, init_a))

    # CEM MPC (GPU batch parallel, K=200, 3 iters)
    cem_times = []
    for trial in range(N_TRIALS):
        ep_s, ep_a = eps_vl[trial % len(eps_vl)]
        ep_sn = (ep_s - m) / (std + 1e-8)
        t = min(trial * 50, len(ep_sn) - T - 1)
        init_s = torch.FloatTensor(ep_sn[t:t+T].reshape(T, -1)).unsqueeze(0).to(device)
        init_a = torch.FloatTensor(ep_a[t:t+T-1].reshape(T-1, -1)).unsqueeze(0).to(device)
        cem_times.append(cem_mpc(model, init_s, init_a))

    g_m, g_s = np.mean(grad_times), np.std(grad_times)
    c_m, c_s = np.mean(cem_times), np.std(cem_times)
    results[model_name] = {
        'gradient_ms': round(g_m, 1), 'gradient_std': round(g_s, 1), 'gradient_hz': round(1000/g_m, 1),
        'cem_ms': round(c_m, 1), 'cem_std': round(c_s, 1), 'cem_hz': round(1000/c_m, 1),
    }
    print(f'  Gradient: {g_m:.1f}±{g_s:.1f}ms ({1000/g_m:.1f}Hz)')
    print(f'  CEM(K={K},3iter): {c_m:.1f}±{c_s:.1f}ms ({1000/c_m:.1f}Hz)')

with open('experiments/cem_mpc_final.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*50)
print('FINAL CEM-MPC RESULTS (GPU batch parallel)')
print('='*50)
for name, r in results.items():
    print(f'{name}: Gradient {r["gradient_ms"]}ms/{r["gradient_hz"]}Hz, CEM {r["cem_ms"]}ms/{r["cem_hz"]}Hz')
print('Done!')
