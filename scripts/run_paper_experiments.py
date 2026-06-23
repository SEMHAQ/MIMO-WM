"""论文所有实验 - 基于run_all_experiments.py结构
包括:
1. 主实验 (已有)
2. 多步预测
3. 消融实验 (D, L, N)
4. 阈值函数对比
5. 超参搜索 (lambda, H)
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel, GRUWorldModel
from src.models.fusion_ssm import FSM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
BS = 256
LR = 5e-4
T = 32

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    eps = []
    for i, f in enumerate(fs):
        eps.append((np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']))
        if (i+1) % 200 == 0: print(f'    {i+1}/{len(fs)}...', flush=True)
    print(f'    {len(fs)} episodes loaded', flush=True)
    return eps

def stats(eps):
    a = np.concatenate([s for s,_ in eps])
    return a.mean(0), a.std(0)

def make_data(eps, T, mean, std):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T+1: continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st)-T, T):
            if j+T >= len(st): break
            Xs.append(sn[j:j+T]); Xa.append(ac[j:j+T-1]); Y.append(sn[j+T])
    return np.array(Xs), np.array(Xa), np.array(Y)

def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device)
    Xav_g = torch.FloatTensor(Xav).to(device)
    Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0; best_ep = ep+1
        else: pat += 1
        if pat >= 20: break
    model.eval()
    with torch.no_grad():
        pred = model(Xv_g, Xav_g)
        mse = loss_fn(pred, Yv_g).item()
        ss_r = torch.sum((Yv_g - pred)**2).item()
        ss_t = torch.sum((Yv_g - torch.mean(Yv_g, dim=0))**2).item()
        r2 = 1 - ss_r / ss_t
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {'mse': round(mse, 6), 'r2': round(r2, 6), 'best_epoch': best_ep, 'params_m': round(params, 3)}

def measure_inference_time(model, Xv, Xav, device):
    """测量推理时间"""
    model.eval()
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 100 * 1000

def multi_step_predict(model, Xv, Xav, Yv, H_list=[1, 4, 8, 16]):
    """多步预测"""
    model.eval()
    results = {}
    for H in H_list:
        mse_h = []
        for i in range(min(50, len(Xv))):
            seq_s = torch.FloatTensor(Xv[i:i+1]).to(device)
            seq_a = torch.FloatTensor(Xav[i:i+1]).to(device)
            true_next = []
            for h in range(H):
                idx_t = i + h
                if idx_t < len(Yv):
                    true_next.append(Yv[idx_t])
            if len(true_next) < H: continue
            preds = []
            cur_s = seq_s.clone()
            cur_a = seq_a.clone()
            for h in range(H):
                with torch.no_grad():
                    p = model(cur_s, cur_a)
                preds.append(p.cpu().numpy()[0])
                cur_s = torch.cat([cur_s[:, 1:], p.unsqueeze(1)], dim=1)
            preds = np.array(preds)
            true_next = np.array(true_next)
            mse_h.append(np.mean((preds - true_next)**2))
        results[f'H{H}'] = round(np.mean(mse_h), 6) if mse_h else None
    return results

# Dataset configs
datasets = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    'ant': {'dir': 'data/ant', 'sd': 105, 'ad': 8},
    'walker2d': {'dir': 'data/walker2d', 'sd': 17, 'ad': 6},
}

# Load data once
print('\n加载数据...', flush=True)
data_cache = {}
for ds_name, ds_cfg in datasets.items():
    print(f'\n{ds_name}:', flush=True)
    eps_tr = load_eps(ds_cfg['dir'], 'train')
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    data_cache[ds_name] = (Xs, Xa, Y, Xv, Xav, Yv, ds_cfg)
    print(f'  Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

# ============================================================
# 实验1: 多步预测 (Humanoid)
# ============================================================
print('\n' + '='*60, flush=True)
print('实验1: 多步预测', flush=True)
print('='*60, flush=True)

multistep_results = {}
Xs, Xa, Y, Xv, Xav, Yv, ds_cfg = data_cache['humanoid']

for model_name in ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']:
    print(f'\n{model_name}:', flush=True)
    multistep_results[model_name] = {}

    for seed in SEEDS:
        print(f'  seed={seed}...', end=' ', flush=True)
        # 训练模型
        if model_name == 'LSTM-WM':
            ModelClass, kwargs = LSTMWorldModel, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'hidden_dim': 128, 'n_layers': 4}
        elif model_name == 'GRU-WM':
            ModelClass, kwargs = GRUWorldModel, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'hidden_dim': 128, 'n_layers': 4}
        elif model_name == 'Transformer-WM':
            ModelClass, kwargs = TransformerWorldModel, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'd_model': 128, 'n_layers': 4}
        elif model_name == 'Mamba-WM':
            ModelClass, kwargs = MambaWorldModel, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'd_model': 128, 'n_layers': 4}
        elif model_name == 'S4D-WM':
            ModelClass, kwargs = SSMWorldModel, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'd_model': 128, 'd_state': 16, 'n_layers': 4}
        elif model_name == 'FSM-WM':
            ModelClass, kwargs = FSM, {'state_dim': ds_cfg['sd'], 'action_dim': ds_cfg['ad'], 'd_model': 128, 'd_state': 16, 'n_layers': 4, 'window_size': 8}

        torch.manual_seed(seed); np.random.seed(seed)
        model = ModelClass(**kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.MSELoss()
        Xv_g = torch.FloatTensor(Xv).to(device)
        Xav_g = torch.FloatTensor(Xav).to(device)
        Yv_g = torch.FloatTensor(Yv).to(device)
        best_val = float('inf'); pat = 0
        for ep in range(EPOCHS):
            model.train()
            idx = np.random.permutation(len(Xs))
            for i in range(0, len(idx), BS):
                bi = idx[i:i+BS]
                pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
                loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sch.step()
            model.eval()
            with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
            if vl < best_val: best_val = vl; pat = 0
            else: pat += 1
            if pat >= 20: break

        # 多步预测
        model.eval()
        multistep = multi_step_predict(model, Xv, Xav, Yv)
        multistep_results[model_name][f'seed{seed}'] = multistep
        print(f'OK', flush=True)

# 保存
with open('experiments/multistep_results.json', 'w') as f:
    json.dump(multistep_results, f, indent=2)

# 打印结果
print('\n多步预测结果:', flush=True)
print(f'{"Model":<16} H1      H4      H8      H16', flush=True)
print('-'*50, flush=True)
for model_name in multistep_results:
    h1 = np.mean([v['H1'] for v in multistep_results[model_name].values() if v['H1']])
    h4 = np.mean([v['H4'] for v in multistep_results[model_name].values() if v['H4']])
    h8 = np.mean([v['H8'] for v in multistep_results[model_name].values() if v['H8']])
    h16 = np.mean([v['H16'] for v in multistep_results[model_name].values() if v['H16']])
    print(f'{model_name:<16} {h1:.3f}  {h4:.3f}  {h8:.3f}  {h16:.3f}', flush=True)

print('\nDone!', flush=True)
