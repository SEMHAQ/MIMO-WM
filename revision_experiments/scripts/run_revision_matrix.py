"""对比矩阵训练 runner —— A1(无门控) / A2-基线(S4D,LRU,Performer) / A3(常规Transformer) / MIMO-WM校准

口径与论文主表(run_exp1_state_prediction.py)完全一致: D=96,N=16,L=2, BS=256, EPOCHS=100,
LR=5e-4, cosine, 早停 pat=20, T=32, seeds [42,123,456,789,1024], Humanoid+HumanoidStandup.
结果按 {model_key_dataset: {seedN: {...}}} 增量写入 results/matrix_results.json(可断点续跑).
仅 MIMO-WM 额外保存 best-val 权重 (供 A9/A10 复用): results/ckpt/{model}_{ds}_seed{seed}.pt

用法(WSL GPU):
  python3 revision_experiments/scripts/run_revision_matrix.py            # 全部
  python3 revision_experiments/scripts/run_revision_matrix.py --jobs MIMO-WM-noGate S4D-WM LRU-WM Performer-WM Transformer-Reg --datasets humanoid
  python3 revision_experiments/scripts/run_revision_matrix.py --jobs MIMO-WM --seeds 42 --only_save  # 快速校准
"""
import sys, os, json, time, math
import torch, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from src.models.ssm_world_model import SSMWorldModel, DiagSSM
from src.models.baselines import LSTMWorldModel, GRUWorldModel, TransformerWorldModel, TCNWorldModel
from src.models.mamba_world_model import MambaWorldModel
from src.models.mimo_world_model import MIMOWorldModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS, BS, LR, T = 100, 256, 5e-4, 32
RESULT = 'revision_experiments/results/matrix_results.json'
CKPT = 'revision_experiments/results/ckpt'
os.makedirs('revision_experiments/results', exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
print(f'Device: {DEVICE}', flush=True)


# ============================================================ 新基线模型
class NoGateMIMO(nn.Module):
    """MIMO-SSM(无门控): 编码器 + (LayerNorm + DiagSSM + 残差)*L + 解码器.
    与 src MIMO-WM 同 encoder/decoder, 仅去掉 output(W_o)*sigmoid(gate) 交互."""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), DiagSSM(d_model, d_state)) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            actions = torch.cat([torch.zeros(states.shape[0], states.shape[1] - actions.shape[1],
                                             actions.shape[-1], device=actions.device), actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for blk in self.backbone:
            h = h + blk(h)
        return states[:, -1] + self.decoder(h[:, -1])


class _LRUBlk(nn.Module):
    """LRU 块: LayerNorm + 对角线性递归 + per-channel γ 输出门 + 残差."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x + self.ssm(self.norm(x)) * torch.sigmoid(self.gamma).unsqueeze(0).unsqueeze(0)


class LRUWorldModel(nn.Module):
    """LRU-WM: 门控对角线性递归(LRU 风格). LayerNorm + DiagSSM + per-channel sigmoid(gamma) 门 + 残差.
    递归核/参数化与 S4D 同为对角线性递归族, 额外带 γ 输出门, 作为"更新对角的递归骨干"对照."""
    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([_LRUBlk(d_model, d_state) for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            actions = torch.cat([torch.zeros(states.shape[0], states.shape[1] - actions.shape[1],
                                             actions.shape[-1], device=actions.device), actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for blk in self.backbone:
            h = blk(h)
        return states[:, -1] + self.decoder(h[:, -1])


class _CausalLinearAttn(nn.Module):
    """因果 FAVOR+ 线性注意力(Performer 式, 随机特征 exp kernel, cumsum 实现因果)."""
    def __init__(self, d, m=384):
        super().__init__()
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d); self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        g = torch.Generator().manual_seed(0)
        w = torch.randn(d, m, generator=g) / math.sqrt(d)
        self.register_buffer('w', w)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)     # (B,L,D)
        fq = torch.exp(q @ self.w)                        # (B,L,m)  查询特征
        fk = torch.exp(k @ self.w)                        # (B,L,m)  键特征
        fkc = fk.cumsum(1)                                # Σ_{τ≤t} φ(k_τ)   (B,L,m)
        kv = (fk.unsqueeze(-1) * v.unsqueeze(2)).cumsum(1)  # Σ_{τ≤t} φ(k_τ)⊗v_τ  (B,L,m,D)
        num = (kv * fq.unsqueeze(-1)).sum(2)              # (B,L,D) = φ(q_t)·Σ_{τ≤t}φ(k_τ)⊗v_τ
        den = (fq * fkc).sum(-1, keepdim=True) + 1e-6     # (B,L,1) = φ(q_t)·Σ_{τ≤t}φ(k_τ)
        return self.wo(num / den)


class PerformerWorldModel(nn.Module):
    """Performer-WM: 因果线性注意力(FAVOR+) 世界模型, 轻量 Attention 变体对照."""
    def __init__(self, state_dim, action_dim, d_model=96, n_layers=2, feat=384):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), _CausalLinearAttn(d_model, feat)) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            actions = torch.cat([torch.zeros(states.shape[0], states.shape[1] - actions.shape[1],
                                             actions.shape[-1], device=actions.device), actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for blk in self.backbone:
            h = h + blk(h)
        return states[:, -1] + self.decoder(h[:, -1])


# ============================================================ 数据
def load_eps(d, s):
    dd = os.path.join(d, s)
    fs = sorted([f for f in os.listdir(dd) if f.endswith('.npz')])
    return [(np.load(os.path.join(dd, f))['states'], np.load(os.path.join(dd, f))['actions']) for f in fs]


def make_data(eps, mean, std, T):
    Xs, Xa, Y = [], [], []
    for st, ac in eps:
        if len(st) < T + 1:
            continue
        sn = (st - mean) / (std + 1e-8)
        for j in range(0, len(st) - T, T):
            if j + T >= len(st):
                break
            Xs.append(sn[j:j + T]); Xa.append(ac[j:j + T - 1]); Y.append(sn[j + T])
    return np.array(Xs), np.array(Xa), np.array(Y)


# ============================================================ 训练
def eval_metrics(model, Xv, Xav, Yv, bs=1024):
    """分块评估: 返回 (mse, r2). 避免大激活模型(如 Performer)在整集前向时 OOM."""
    model.eval()
    n = Xv.shape[0]
    ss_err = 0.0; ss_tot = 0.0
    mean_y = Yv.mean(0)
    with torch.no_grad():
        for i in range(0, n, bs):
            p = model(Xv[i:i + bs], Xav[i:i + bs])
            ss_err += ((Yv[i:i + bs] - p) ** 2).sum().item()
            ss_tot += ((Yv[i:i + bs] - mean_y) ** 2).sum().item()
    nelem = n * Yv.shape[1]          # 样本数 × 状态维
    mse = ss_err / nelem
    r2 = 1 - ss_err / ss_tot
    return mse, r2


def train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, save_ckpt=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModelClass(**kwargs).to(DEVICE)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(DEVICE); Xav_g = torch.FloatTensor(Xav).to(DEVICE); Yv_g = torch.FloatTensor(Yv).to(DEVICE)
    best_val = float('inf'); pat = 0; best_ep = 0; best_sd = None
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i + BS]
            pred = model(torch.FloatTensor(Xs[bi]).to(DEVICE), torch.FloatTensor(Xa[bi]).to(DEVICE))
            loss = loss_fn(pred, torch.FloatTensor(Y[bi]).to(DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        vl, _ = eval_metrics(model, Xv_g, Xav_g, Yv_g)
        if vl < best_val:
            best_val = vl; pat = 0; best_ep = ep + 1
            if save_ckpt:
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 20:
            break
    mse, r2 = eval_metrics(model, Xv_g, Xav_g, Yv_g)
    xd = torch.FloatTensor(Xv[:1]).to(DEVICE); ad = torch.FloatTensor(Xav[:1]).to(DEVICE)
    with torch.no_grad():
        for _ in range(5): model(xd, ad)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): model(xd, ad)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        inf_time = (time.perf_counter() - t0) / 100 * 1000
    if save_ckpt:
        torch.save(best_sd, save_ckpt)
    return {'mse': round(mse, 6), 'r2': round(r2, 4), 'params_m': round(params, 3),
            'inf_time_ms': round(inf_time, 2), 'best_epoch': best_ep}


# ============================================================ 模型注册
DATASETS = {
    'humanoid': {'dir': 'data/humanoid', 'sd': 348, 'ad': 17},
    'humanoid_standup': {'dir': 'data/humanoid_standup', 'sd': 348, 'ad': 17},
}

def models_reg():
    def t(kw): return (lambda sd, ad: dict(kw, state_dim=sd, action_dim=ad))
    return {
        'MIMO-WM': (MIMOWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
        'MIMO-WM-noGate': (NoGateMIMO, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),     # A1
        'S4D-WM': (SSMWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),           # A2
        'LRU-WM': (LRUWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),           # A2
        'Performer-WM': (PerformerWorldModel, t({'d_model': 96, 'n_layers': 2, 'feat': 192})),  # A4 (m=2D 控显存)
        'Transformer-Reg': (TransformerWorldModel, t({'d_model': 192, 'nhead': 6, 'n_layers': 3})),  # A3 常规规模
    }


if __name__ == '__main__':
    argv = sys.argv[1:]
    jobs = None; dss = None; seeds = SEEDS
    if '--jobs' in argv:
        jobs = argv[argv.index('--jobs') + 1:argv.index('--jobs') + 7]
    if '--datasets' in argv:
        dss = argv[argv.index('--datasets') + 1:argv.index('--datasets') + 3]
    if '--seeds' in argv:
        seeds = [int(s) for s in argv[argv.index('--seeds') + 1:argv.index('--seeds') + 6]]
    reg = models_reg()
    if jobs is None:
        jobs = list(reg.keys())
    if dss is None:
        dss = list(DATASETS.keys())

    results = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    for ds_name in dss:
        cfg = DATASETS[ds_name]
        print(f'\n加载 {ds_name} ...', flush=True)
        eps_tr = load_eps(cfg['dir'], 'train'); eps_vl = load_eps(cfg['dir'], 'val')
        a = np.concatenate([s for s, _ in eps_tr]); m, s = a.mean(0), a.std(0)
        Xs, Xa, Y = make_data(eps_tr, m, s, T)
        Xv, Xav, Yv = make_data(eps_vl, m, s, T)
        print(f'  Train {len(Xs)}  Val {len(Xv)}', flush=True)
        for name in jobs:
            key = f'{name}_{ds_name}'
            if key in results and len(results[key]) >= len(seeds):
                print(f'{key}: 完整, 跳过', flush=True); continue
            results.setdefault(key, {})
            ModelClass, kwfn = reg[name]
            kwargs = kwfn(cfg['sd'], cfg['ad'])
            for seed in seeds:
                sk = f'seed{seed}'
                if sk in results[key]:
                    continue
                ck = os.path.join(CKPT, f'{name}_{ds_name}_seed{seed}.pt') if name == 'MIMO-WM' else None
                r = train_eval(ModelClass, kwargs, Xs, Xa, Y, Xv, Xav, Yv, seed, save_ckpt=ck)
                results[key][sk] = r
                print(f'  {key} {sk}: MSE={r["mse"]:.4f} R2={r["r2"]:.4f} P={r["params_m"]}M ep={r["best_epoch"]}', flush=True)
                with open(RESULT, 'w') as f:
                    json.dump(results, f, indent=2)
    # 汇总
    print('\n=== 汇总(mean±std MSE) ===', flush=True)
    for ds_name in dss:
        print(f'[{ds_name}]', flush=True)
        for name in jobs:
            key = f'{name}_{ds_name}'
            if key in results:
                v = [results[key][s]['mse'] for s in results[key] if 'mse' in results[key][s]]
                if v:
                    print(f'  {name:<18} {np.mean(v)*100:.2f}±{np.std(v)*100:.2f}  ({results[key]["seed42"]["params_m"]}M)', flush=True)
    print('\nDone!', flush=True)
