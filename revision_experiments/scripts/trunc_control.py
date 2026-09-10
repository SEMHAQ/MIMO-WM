"""A10: 长序列退化机理·截断对照 —— 训练 MIMO-WM at T=128, 评估"全 128 输入 vs 仅末尾 32 输入".
用途:若截断不损失精度 => 退化源于远端冗余上下文稀释, 而非模型不会用长上下文.

需 GPU; 运行(WSL):
  python3 revision_experiments/scripts/trunc_control.py [--seeds 42 123]
产物: revision_experiments/results/trunc_control.json
"""
import os, sys, json, torch, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SD, AD, TLONG = 348, 17, 128
EPOCHS, BS, LR = 100, 256, 5e-4


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


def train(seed, Xs, Xa, Y, Xv, Xav, Yv):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MIMOWorldModel(SD, AD, d_model=96, d_state=16, n_layers=2).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    fn = nn.MSELoss()
    Xv_g = torch.FloatTensor(Xv).to(DEV); Xav_g = torch.FloatTensor(Xav).to(DEV); Yv_g = torch.FloatTensor(Yv).to(DEV)
    best = float('inf'); pat = 0; best_sd = None
    for ep in range(EPOCHS):
        m.train()
        idx = np.random.permutation(len(Xs))
        for i in range(0, len(idx), BS):
            bi = idx[i:i + BS]
            loss = fn(m(torch.FloatTensor(Xs[bi]).to(DEV), torch.FloatTensor(Xa[bi]).to(DEV)), torch.FloatTensor(Y[bi]).to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = fn(m(Xv_g, Xav_g), Yv_g).item()
        if vl < best:
            best = vl; pat = 0; best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            pat += 1
        if pat >= 20:
            break
    m.load_state_dict(best_sd)
    return m


def main():
    global EPOCHS
    seeds = [int(x) for x in sys.argv[sys.argv.index('--seeds') + 1:sys.argv.index('--seeds') + 3]] if '--seeds' in sys.argv else [42]
    if '--epochs' in sys.argv:
        EPOCHS = int(sys.argv[sys.argv.index('--epochs') + 1])
    eps_tr = load_eps('data/humanoid', 'train'); eps_vl = load_eps('data/humanoid', 'val')
    a = np.concatenate([s for s, _ in eps_tr]); mean, std = a.mean(0), a.std(0)
    Xs, Xa, Y = make_data(eps_tr, mean, std, TLONG)
    Xv, Xav, Yv = make_data(eps_vl, mean, std, TLONG)
    res = {}
    for seed in seeds:
        m = train(seed, Xs, Xa, Y, Xv, Xav, Yv).eval()
        m.to('cpu')
        Xg = torch.FloatTensor(Xv[:500]); Xag = torch.FloatTensor(Xav[:500]); Yg = torch.FloatTensor(Yv[:500])
        with torch.no_grad():
            pred_full = m(Xg, Xag)
            mse_full = nn.functional.mse_loss(pred_full, Yg).item()
            pred_trunc = m(Xg[:, -32:], Xag[:, -31:])   # 仅末尾 32 步(动作 31 步)
            mse_trunc = nn.functional.mse_loss(pred_trunc, Yg).item()
        res[seed] = {'mse_full_T128': round(mse_full, 6), 'mse_trunc_tail32': round(mse_trunc, 6),
                     'loss_ratio_trunc_over_full': round(mse_trunc / mse_full, 4)}
        print(f'seed {seed}: full(128) MSE={mse_full:.4f} | tail-32 MSE={mse_trunc:.4f} | ratio={mse_trunc/mse_full:.3f}', flush=True)
    out = 'trunc_longbudget.json' if ('--epochs' in sys.argv) else 'trunc_control.json'
    json.dump(res, open('revision_experiments/results/' + out, 'w'), indent=2)
    print(f'Saved -> revision_experiments/results/{out}', flush=True)


if __name__ == '__main__':
    main()
