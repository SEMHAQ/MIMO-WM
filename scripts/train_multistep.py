"""Train S4D-WM with TRUE multi-step rollout loss on D4RL Humanoid.
Uses autoregressive rollout during training to optimize multi-step prediction.
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel

device = torch.device('cuda')
STATE_DIM = 348; ACTION_DIM = 17; T = 32; BS = 64; LR = 5e-4; SEED = 42

def load_episodes(data_dir, split, max_eps=None):
    d = os.path.join(data_dir, split)
    files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
    if max_eps: files = files[:max_eps]
    episodes = []
    for f in files:
        d2 = np.load(os.path.join(d, f))
        episodes.append((d2['states'], d2['actions']))
    return episodes

def make_multistep_data(episodes, T, H, mean, std):
    """Create data for multi-step rollout training.
    Input: states[t:t+T], actions[t:t+T-1]
    Target: states[t+T:t+T+H] (next H steps)
    """
    Xs, Xa, Ys = [], [], []
    for st, ac in episodes:
        if len(st) < T + H: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st) - T - H + 1, T):
            if j + T + H > len(st): break
            Xs.append(st_n[j:j+T])
            Xa.append(ac[j:j+T-1])
            Ys.append(st_n[j+T:j+T+H])
    return np.array(Xs), np.array(Xa), np.array(Ys)

def compute_stats(episodes):
    all_states = np.concatenate([st for st, _ in episodes], axis=0)
    return all_states.mean(axis=0), all_states.std(axis=0)

def train_multistep(model, name, eps_tr, eps_vl, mean, std, 
                    T=32, H=8, lam=0.5, epochs=100, lr=5e-4, bs=64):
    """Train with combined single-step + multi-step rollout loss."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    # Single-step data
    Xs1, Xa1, Y1 = [], [], []
    for st, ac in eps_tr:
        if len(st) < T+1: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs1.append(st_n[j:j+T]); Xa1.append(ac[j:j+T-1]); Y1.append(st_n[j+T])
    Xs1, Xa1, Y1 = np.array(Xs1), np.array(Xa1), np.array(Y1)
    
    Xv1, Xav1, Yv1 = [], [], []
    for st, ac in eps_vl:
        if len(st) < T+1: continue
        st_n = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xv1.append(st_n[j:j+T]); Xav1.append(ac[j:j+T-1]); Yv1.append(st_n[j+T])
    Xv1, Xav1, Yv1 = np.array(Xv1), np.array(Xav1), np.array(Yv1)
    
    # Multi-step data (smaller subset to save memory)
    XsH, XaH, YH = make_multistep_data(eps_tr[:300], T, H, mean, std)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  {name}: {params:.2f}M params, H={H}, lambda={lam}', flush=True)
    print(f'  Single-step: train={len(Xs1)}, val={len(Xv1)}', flush=True)
    print(f'  Multi-step: train={len(XsH)}', flush=True)
    
    best_val = float('inf')
    patience_counter = 0
    epoch_logs = []
    
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs1))
        losses = []
        
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xs = torch.FloatTensor(Xs1[bi]).to(device)
            xa = torch.FloatTensor(Xa1[bi]).to(device)
            y = torch.FloatTensor(Y1[bi]).to(device)
            
            # Single-step loss
            pred = model(xs, xa)
            loss_single = loss_fn(pred, y)
            
            # Multi-step rollout loss
            if lam > 0 and len(XsH) > 0:
                # Sample multi-step batch
                mbi = np.random.choice(len(XsH), min(bs, len(XsH)), replace=False)
                xs_m = torch.FloatTensor(XsH[mbi]).to(device)
                xa_m = torch.FloatTensor(XaH[mbi]).to(device)
                ys_m = torch.FloatTensor(YH[mbi]).to(device)  # (bs, H, state_dim)
                
                # Autoregressive rollout
                loss_ms = 0.0
                current_states = xs_m.clone()
                current_actions = xa_m.clone()
                
                for h in range(H):
                    pred_h = model(current_states, current_actions)
                    loss_ms += loss_fn(pred_h, ys_m[:, h, :])
                    
                    # Shift window for next step
                    if h < H - 1:
                        # Append prediction, remove first state
                        new_states = torch.cat([current_states[:, 1:, :], pred_h.unsqueeze(1)], dim=1)
                        # Get next action from ground truth (teacher forcing for actions)
                        if current_actions.shape[1] >= T - 1:
                            new_actions = torch.cat([current_actions[:, 1:, :], 
                                                    torch.zeros(xs_m.shape[0], 1, ACTION_DIM).to(device)], dim=1)
                        else:
                            new_actions = current_actions
                        current_states = new_states
                        current_actions = new_actions
                
                loss_ms = loss_ms / H
                loss = (1 - lam) * loss_single + lam * loss_ms
            else:
                loss = loss_single
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sch.step()
        
        # Validate single-step
        model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(Xv1).to(device)
            xav = torch.FloatTensor(Xav1).to(device)
            yv = torch.FloatTensor(Yv1).to(device)
            val_loss = loss_fn(model(xv, xav), yv).item()
        
        train_loss = np.mean(losses)
        epoch_logs.append({'epoch': ep+1, 'train_loss': train_loss, 'val_loss': val_loss})
        
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'experiments/{name}.pth')
        else:
            patience_counter += 1
        
        if (ep+1) % 20 == 0 or ep == 0:
            print(f'    Epoch {ep+1}: train={train_loss:.6f}, val={val_loss:.6f}', flush=True)
        
        if patience_counter >= 20:
            print(f'    Early stop at epoch {ep+1}', flush=True)
            break
    
    print(f'  {name} done: best_val={best_val:.6f}', flush=True)
    return best_val, epoch_logs

def evaluate_multistep(model_name, kwargs, eps_vl, mean, std, T=32, horizons=[1,4,8,16]):
    """Evaluate multi-step rollout."""
    model = SSMWorldModel(**kwargs).to(device)
    model.load_state_dict(torch.load(f'experiments/{model_name}.pth', map_location=device))
    model.eval()
    
    episodes = eps_vl[:50]
    results = {}
    
    for H in horizons:
        mse_list = []
        for st, ac in episodes:
            if len(st) < T + H: continue
            st_n = (st - mean) / (std + 1e-8)
            for j in range(0, len(st) - T - H, T):
                if j + T + H > len(st): break
                states_in = st_n[j:j+T].copy()
                actions_in = ac[j:j+T-1].copy()
                rollout_preds = []
                
                for h in range(H):
                    xs = torch.FloatTensor(states_in).unsqueeze(0).to(device)
                    xa = torch.FloatTensor(actions_in).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = model(xs, xa).cpu().numpy()[0]
                    rollout_preds.append(pred)
                    if j + T + h < len(ac):
                        states_in = np.concatenate([states_in[1:], pred.reshape(1, -1)])
                        next_action = ac[j + T + h].reshape(1, -1)
                        actions_in = np.concatenate([actions_in[1:], next_action])
                
                gt = st_n[j + T + H - 1]
                mse_list.append(np.mean((rollout_preds[-1] - gt) ** 2))
        
        results[H] = np.mean(mse_list) if mse_list else float('inf')
        print(f'    H={H}: MSE={results[H]:.6f} (n={len(mse_list)})', flush=True)
    
    return results

if __name__ == '__main__':
    print("="*60, flush=True)
    print("S4D-WM Multi-step Training on D4RL Humanoid", flush=True)
    print("="*60, flush=True)
    
    eps_tr = load_episodes('data/humanoid', 'train', 930)
    eps_vl = load_episodes('data/humanoid', 'val', 233)
    mean, std = compute_stats(eps_tr)
    
    kwargs = {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'd_model': 128, 'd_state': 16, 'n_layers': 4}
    
    results = {}
    
    # Config 1: λ=0.5, H=8 (balanced)
    print("\n--- Config 1: λ=0.5, H=8 ---", flush=True)
    model = SSMWorldModel(**kwargs)
    train_multistep(model, 'S4D-WM_ms_H8', eps_tr, eps_vl, mean, std, H=8, lam=0.5, epochs=100)
    ms_h8 = evaluate_multistep('S4D-WM_ms_H8', kwargs, eps_vl, mean, std)
    results['H8_lam05'] = ms_h8
    
    # Config 2: λ=0.8, H=16 (more emphasis on multi-step)
    print("\n--- Config 2: λ=0.8, H=16 ---", flush=True)
    model = SSMWorldModel(**kwargs)
    train_multistep(model, 'S4D-WM_ms_H16', eps_tr, eps_vl, mean, std, H=16, lam=0.8, epochs=100)
    ms_h16 = evaluate_multistep('S4D-WM_ms_H16', kwargs, eps_vl, mean, std)
    results['H16_lam08'] = ms_h16
    
    # Config 3: λ=0.3, H=4 (light multi-step)
    print("\n--- Config 3: λ=0.3, H=4 ---", flush=True)
    model = SSMWorldModel(**kwargs)
    train_multistep(model, 'S4D-WM_ms_H4', eps_tr, eps_vl, mean, std, H=4, lam=0.3, epochs=100)
    ms_h4 = evaluate_multistep('S4D-WM_ms_H4', kwargs, eps_vl, mean, std)
    results['H4_lam03'] = ms_h4
    
    print("\n" + "="*60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    for cfg, ms in results.items():
        print(f"{cfg}: {ms}", flush=True)
    
    with open('experiments/multistep_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/multistep_training_results.json", flush=True)
