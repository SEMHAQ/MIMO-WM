"""基线模型实验 - LSTM, GRU, Transformer, Mamba, S4D
与新模型探索脚本使用相同配置，确保公平对比
"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPOCHS = 100
BS = 256
T = 32

print(f'Device: {device}', flush=True)

def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]

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

def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=SEED):
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = ModelClass(**kwargs)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0; best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i+BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
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

    # 推理时间
    with torch.no_grad():
        x_dummy = torch.FloatTensor(Xv[:1]).to(device)
        a_dummy = torch.FloatTensor(Xav[:1]).to(device)
        for _ in range(5): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(x_dummy, a_dummy)
        torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000

    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params_m': round(params, 3), 'inf_time_ms': round(inf_time, 2), 'best_epoch': best_ep}

# 基线模型配置（与新模型探索相同：d_model=96, n_layers=2）
configs = {
    'LSTM': (LSTMWorldModel, {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
    'GRU': (GRUWorldModel, {'state_dim': 348, 'action_dim': 17, 'hidden_dim': 96, 'n_layers': 2}),
    'Transformer': (TransformerWorldModel, {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'nhead': 4, 'n_layers': 2}),
    'Mamba': (MambaWorldModel, {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'n_layers': 2}),
    'S4D': (SSMWorldModel, {'state_dim': 348, 'action_dim': 17, 'd_model': 96, 'd_state': 16, 'n_layers': 2}),
}

if __name__ == '__main__':
    print('\n加载Humanoid数据...', flush=True)
    eps_tr = load_eps('data/humanoid', 'train')
    eps_vl = load_eps('data/humanoid', 'val')
    m, s = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, m, s)
    Xv, Xav, Yv = make_data(eps_vl, T, m, s)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    RESULTS_FILE = 'experiments/baselines.json'
    os.makedirs('experiments', exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    print('\n' + '='*60, flush=True)
    print('基线模型实验', flush=True)
    print('='*60, flush=True)

    for name, (ModelClass, kwargs) in configs.items():
        if name in results:
            print(f'\n{name}: 已有结果，跳过', flush=True)
            continue

        print(f'\n{name}:', flush=True)
        r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv)
        results[name] = r
        print(f'  MSE={r["mse"]:.4f}, R²={r["r2"]:.4f}, Params={r["params_m"]:.3f}M, Infer={r["inf_time_ms"]:.2f}ms', flush=True)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # 打印结果
    print('\n' + '='*60, flush=True)
    print('基线模型结果', flush=True)
    print('='*60, flush=True)
    print('{:<16} {:<10} {:<10} {:<10} {:<10}'.format('模型', 'MSE', 'R²', '参数(M)', '推理(ms)'))
    print('-'*60)

    for name in configs:
        if name in results:
            r = results[name]
            print('{:<16} {:<10.4f} {:<10.4f} {:<10.3f} {:<10.2f}'.format(
                name, r['mse'], r['r2'], r['params_m'], r['inf_time_ms']
            ))

    print('\nDone!', flush=True)
