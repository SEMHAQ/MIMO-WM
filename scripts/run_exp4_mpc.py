"""实验4：MPC规划 — 速度 + 闭环控制质量"""
import torch, torch.nn as nn, numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel, TCNWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.mimo_world_model import MIMOWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# ============ 梯度MPC ============
class GradientMPC:
    def __init__(self, model, horizon=10, n_iter=30, lr=0.01, Q=1.0, R=0.01):
        self.model = model; self.H = horizon; self.n_iter = n_iter; self.lr = lr; self.Q = Q; self.R = R
    def plan(self, state_hist, action_hist, target):
        a_dim = action_hist.shape[1]
        a_seq = nn.Parameter(torch.randn(self.H, a_dim, device=device)*0.1)
        opt = torch.optim.Adam([a_seq], lr=self.lr)
        s_h = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_h = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)
        was_eval = not self.model.training
        if was_eval: self.model.train()
        for _ in range(self.n_iter):
            opt.zero_grad()
            cost = torch.tensor(0.0, device=device)
            s, a = s_h.clone(), a_h.clone()
            for h in range(self.H):
                pred = self.model(s, a)
                cost = cost + self.Q * torch.norm(pred-tgt) + self.R * torch.norm(a_seq[h])
                s = torch.cat([s[:,1:], pred.unsqueeze(1)], dim=1)
                a = torch.cat([a[:,1:], a_seq[h:h+1].unsqueeze(0)], dim=1)
            cost.backward(); opt.step()
        if was_eval: self.model.eval()
        return a_seq[0].detach().cpu().numpy()

# ============ CEM-MPC ============
class CEMMPC:
    def __init__(self, model, horizon=10, n_samples=256, n_elite=32, n_iter=5, Q=1.0, R=0.01):
        self.model = model; self.H = horizon; self.n_samples = n_samples
        self.n_elite = n_elite; self.n_iter = n_iter; self.Q = Q; self.R = R
    def plan(self, state_hist, action_hist, target):
        a_dim = action_hist.shape[1]
        s_h = torch.FloatTensor(state_hist).unsqueeze(0).to(device)
        a_h = torch.FloatTensor(action_hist).unsqueeze(0).to(device)
        tgt = torch.FloatTensor(target).to(device)
        mu = torch.zeros(self.H, a_dim, device=device)
        sigma = torch.ones(self.H, a_dim, device=device)
        for _ in range(self.n_iter):
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(self.n_samples, self.H, a_dim, device=device)
            samples = samples.clamp(-1, 1)
            costs = torch.zeros(self.n_samples, device=device)
            s_cur = s_h.expand(self.n_samples, -1, -1).clone()
            a_cur = a_h.expand(self.n_samples, -1, -1).clone()
            for h in range(self.H):
                pred = self.model(s_cur, a_cur)
                costs = costs + self.Q * torch.norm(pred - tgt.unsqueeze(0), p=2, dim=-1) + self.R * torch.norm(samples[:,h], p=2, dim=-1)
                s_cur = torch.cat([s_cur[:,1:], pred.unsqueeze(1)], dim=1)
                a_cur = torch.cat([a_cur[:,1:], samples[:,h:h+1]], dim=1)
            elite = samples[costs.topk(self.n_elite, largest=False).indices]
            mu = elite.mean(dim=0); sigma = elite.std(dim=0) + 1e-6
        return mu[0].detach().cpu().numpy()

# ============ 训练 ============
def train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(device); Xav_g = torch.FloatTensor(Xav).to(device); Yv_g = torch.FloatTensor(Yv).to(device)
    best_val = float('inf'); pat = 0
    for ep in range(100):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            pred = model(torch.FloatTensor(Xs[bi]).to(device), torch.FloatTensor(Xa[bi]).to(device))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xv_g, Xav_g), Yv_g).item()
        if vl < best_val: best_val = vl; pat = 0
        else: pat += 1
        if pat >= 20: break
    return model

# ============ 速度评测 ============
def eval_speed(mpc, episodes, mean, std, n_episodes=10, n_steps=50):
    total_time = 0; count = 0
    for states, actions in episodes[:n_episodes]:
        if len(states) < T + n_steps: continue
        s_norm = (states - mean) / (std + 1e-8)
        target = s_norm[T]
        t0 = time.perf_counter()
        for step in range(n_steps):
            s_hist = s_norm[step:step+T]; a_hist = actions[step:step+T-1]
            mpc.plan(s_hist, a_hist, target)
            if T + step + 1 < len(s_norm): target = s_norm[T + step + 1]
        total_time += time.perf_counter() - t0; count += 1
    if count == 0: return {'hz': 0, 'avg_step_time_ms': 0}
    avg = total_time / count / n_steps
    return {'hz': round(1.0/avg, 2) if avg>0 else 0, 'avg_step_time_ms': round(avg*1000, 2)}

# ============ 多步预测精度评测（MPC规划质量指标）============
def eval_quality(model, episodes, mean, std, n_episodes=5, H=3):
    """多步开环预测精度：MPC规划所需的rollout质量, 直接决定控制性能"""
    model.eval()
    errors = []
    for states, actions in episodes[:n_episodes]:
        if len(states) < T + H: continue
        s_norm = (states - mean) / (std + 1e-8)
        s_hist = s_norm[:T].copy()
        a_hist = actions[:T-1].copy()
        real_future = s_norm[T:T+H]
        future_actions = actions[T-1:T-1+H]

        preds = []
        s_cur = s_hist.copy()
        a_cur = a_hist.copy()
        for h in range(H):
            with torch.no_grad():
                s_t = torch.FloatTensor(s_cur).unsqueeze(0).to(device)
                a_t = torch.FloatTensor(a_cur).unsqueeze(0).to(device)
                pred = model(s_t, a_t).cpu().numpy().flatten()
            preds.append(pred)
            s_cur = np.vstack([s_cur[1:], pred.reshape(1, -1)])
            a_cur = np.vstack([a_cur[1:], future_actions[h].reshape(1, -1)])

        preds = np.array(preds)
        step_errors = np.mean((preds - real_future) ** 2, axis=1)
        errors.append(np.mean(step_errors))

    return round(float(np.mean(errors)), 6) if errors else 0

# ============ 主实验 ============
if __name__ == '__main__':
    ds_cfg = {'dir': 'data/humanoid', 'sd': 348, 'ad': 17}
    models = {
        'LSTM-WM': (LSTMWorldModel, {'state_dim':348,'action_dim':17,'hidden_dim':96,'n_layers':2}),
        'GRU-WM': (GRUWorldModel, {'state_dim':348,'action_dim':17,'hidden_dim':96,'n_layers':2}),
        'Transformer-WM': (TransformerWorldModel, {'state_dim':348,'action_dim':17,'d_model':96,'nhead':4,'n_layers':2}),
        'Mamba-WM': (MambaWorldModel, {'state_dim':348,'action_dim':17,'d_model':96,'n_layers':2}),
        'TCN-WM': (TCNWorldModel, {'state_dim':348,'action_dim':17,'d_model':96,'n_layers':2,'kernel_size':3}),
        'MIMO-WM': (MIMOWorldModel, {'state_dim':348,'action_dim':17,'d_model':96,'d_state':16,'n_layers':2}),
    }

    RESULTS_FILE = 'experiments/exp4_mpc.json'
    os.makedirs('experiments', exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    eps_tr = load_eps(ds_cfg['dir'], 'train')
    eps_vl = load_eps(ds_cfg['dir'], 'val')
    mean, std = stats(eps_tr)
    Xs, Xa, Y = make_data(eps_tr, T, mean, std)
    Xv, Xav, Yv = make_data(eps_vl, T, mean, std)
    print(f'Train: {len(Xs)}, Val: {len(Xv)}', flush=True)

    for model_name, (ModelClass, kwargs) in models.items():
        g_key = f'{model_name}_GradMPC'
        c_key = f'{model_name}_CEMMPC'
        q_key = f'{model_name}_Quality'
        if g_key in results and c_key in results and q_key in results:
            print(f'\n[{model_name}]: 已有结果，跳过', flush=True)
            continue

        print(f'\n[{model_name}]', flush=True)
        print(f'  训练...', flush=True)
        model = train_model(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv)
        model.eval()

        if g_key not in results:
            print(f'  梯度MPC速度...', flush=True)
            grad = GradientMPC(model, horizon=10, n_iter=30, lr=0.01)
            r = eval_speed(grad, eps_vl, mean, std)
            results[g_key] = r
            print(f'    {r["hz"]:.2f} Hz', flush=True)

        if c_key not in results:
            print(f'  CEM-MPC速度...', flush=True)
            cem = CEMMPC(model, horizon=10, n_samples=256, n_elite=32, n_iter=5)
            r = eval_speed(cem, eps_vl, mean, std)
            results[c_key] = r
            print(f'    {r["hz"]:.2f} Hz', flush=True)

        if q_key not in results:
            print(f'  多步预测质量...', flush=True)
            q = eval_quality(model, eps_vl, mean, std)
            results[q_key] = q
            print(f'    prediction MSE = {q:.6f}', flush=True)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '='*80)
    print('MPC规划结果')
    print('='*80)
    print(f'{"模型":<18} {"Grad Hz":<10} {"CEM Hz":<10} {"预测MSE":<12}')
    print('-'*55)
    for m in models:
        g = results.get(f'{m}_GradMPC', {}).get('hz', 0)
        c = results.get(f'{m}_CEMMPC', {}).get('hz', 0)
        q = results.get(f'{m}_Quality', 0)
        print(f'{m:<18} {g:<10.2f} {c:<10.2f} {q:<12.6f}')
    print('\nDone!', flush=True)
