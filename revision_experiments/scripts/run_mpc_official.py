"""在 MPC 口径下重跑表 5 中 S4D-WM 与 LRU-WM 两行的官方参考实现。

表 1、表 2 中这两行已改用官方参考实现（见 run_official_s4d.py / run_official_lru.py），
而表 5 的 MPC 行仍取自旧的非本尊实现，两者口径不一致，故在此按同一 MPC 协议重跑。

训练与 MPC 评测协议完全复用 mpc_extended.py：BS=1024、100 epochs、早停 pat=20、
seed=42、CEM 参数 K=256 / Ne=32 / M=5，只替换骨干实现。同时重跑控制组
（原实现 S4D、原实现 LRU），用于核对本脚本能否复现 mpc_extended.json 的数值 ——
控制组对不上说明管线有问题，官方行的结果也不可信。

结果写入 revision_experiments/results/mpc_official.json（原子写入，可断点续跑）。

运行（WSL GPU，工作目录为仓库根）:
  python3 -u revision_experiments/scripts/run_mpc_official.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

import mpc_extended as MX
from run_revision_matrix import NoGateMIMO, LRUWorldModel, SSMWorldModel
from run_official_s4d import OfficialS4DWorldModel
from run_official_lru import OfficialLRUWorldModel

OUT = 'revision_experiments/results/mpc_official.json'
SD, AD = 348, 17
KW = dict(state_dim=SD, action_dim=AD, d_model=96, n_layers=2)

# 先跑控制组：几分钟内即可判定管线是否可信，再决定是否值得等官方 LRU 的数小时
MODELS = {
    'S4D-WM(原实现)': (SSMWorldModel,           dict(KW, d_state=16)),
    'LRU-WM(原实现)': (LRUWorldModel,           dict(KW, d_state=16)),
    'S4D-WM(官方)':   (OfficialS4DWorldModel,   dict(KW, d_state=16)),
    'LRU-WM(官方)':   (OfficialLRUWorldModel,   dict(KW, d_state=16)),
}


def train_with_oom_fallback(MC, kw, data):
    """mpc_extended 用 BS=1024；官方 LRU 的递归扫描在 BS=1024 下可能显存不足，逐级回退。"""
    Xs, Xa, Y, Xv, Xav, Yv = data
    for bs in (MX.BS, 512, 256):
        try:
            return MX.train_model(MC, kw, Xs, Xa, Y, Xv, Xav, Yv, bs=bs), bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f'    BS={bs} 显存不足, 回退', flush=True)
    raise RuntimeError('BS=256 仍显存不足')


if __name__ == '__main__':
    eps_tr = MX.load_eps('data/humanoid', 'train')
    eps_vl = MX.load_eps('data/humanoid', 'val')
    a = np.concatenate([s for s, _ in eps_tr]); mean, std = a.mean(0), a.std(0)
    data = (*MX.make_data(eps_tr, mean, std), *MX.make_data(eps_vl, mean, std))
    print(f'Train {len(data[0])} Val {len(data[3])}', flush=True)

    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

    for name, (MC, kw) in MODELS.items():
        gk, ck, qk = f'{name}_GradMPC', f'{name}_CEMMPC', f'{name}_Quality'
        if all(k in res for k in (gk, ck, qk)):
            print(f'{name}: 已有结果, 跳过', flush=True); continue
        print(f'\n[{name}] 训练...', flush=True)
        m, bs_used = train_with_oom_fallback(MC, kw, data)
        res[f'{name}_bs'] = bs_used
        if bs_used != MX.BS:
            print(f'    实际训练批量大小 {bs_used}', flush=True)
        if gk not in res:
            print('  梯度 MPC...', flush=True); res[gk] = MX.eval_speed(MX.GradientMPC(m), eps_vl, mean, std)
            print('   ', res[gk], flush=True)
        if ck not in res:
            print('  CEM MPC...', flush=True); res[ck] = MX.eval_speed(MX.CEMMPC(m), eps_vl, mean, std)
            print('   ', res[ck], flush=True)
        if qk not in res:
            res[qk] = MX.eval_quality(m, eps_vl, mean, std)
            print('   3步 MSE =', res[qk], flush=True)
        with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        os.replace(OUT + '.tmp', OUT)

    print('\n=== MPC 官方实现结果 ===', flush=True)
    for name in MODELS:
        g = res.get(f'{name}_GradMPC', {}).get('hz', 0)
        c = res.get(f'{name}_CEMMPC', {}).get('hz', 0)
        q = res.get(f'{name}_Quality', 0)
        print(f'{name:<16} Grad {g:>6} Hz | CEM {c:>6} Hz | 3步MSE {q}', flush=True)
    print('\nDone!', flush=True)
