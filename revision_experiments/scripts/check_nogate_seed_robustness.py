# -*- coding: utf-8 -*-
"""核对表 2 中无门控变体在 HumanoidStandup 上的 50.61 是否为种子运气。

改稿时新增的 w/o 门控一行在 Standup 上优于完整模型（50.61 对 53.10），差距 2.5，
远超 5 个种子的标准差（0.22），故须排除"挑种子"的可能。本脚本在其余配置与表 2
完全一致的前提下，补 5 组从未用过的种子重跑 NoGateMIMO。

结果（2026-09-15）:
  原 5 seed  50.61 ± 0.22
  新 5 seed  50.36 ± 0.14
  合 10 seed 50.48 ± 0.22
即该优势不是种子运气，10 种子口径下完整模型 53.10±0.07 仍明显更差。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))
import run_revision_matrix as R

OUT = 'revision_experiments/results/nogate_seed_robustness.json'
NEW_SEEDS = [7, 2024, 31337, 555, 8888]
OLD_SEEDS = [42, 123, 456, 789, 1024]

res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

cfg = R.DATASETS['humanoid_standup']
print('加载 humanoid_standup ...', flush=True)
eps_tr = R.load_eps(cfg['dir'], 'train')
eps_vl = R.load_eps(cfg['dir'], 'val')
a = np.concatenate([s for s, _ in eps_tr])
m, s = a.mean(0), a.std(0)
Xs, Xa, Y = R.make_data(eps_tr, m, s, R.T)
Xv, Xav, Yv = R.make_data(eps_vl, m, s, R.T)
print(f'  Train {len(Xs)}  Val {len(Xv)}', flush=True)

kw = dict(state_dim=cfg['sd'], action_dim=cfg['ad'], d_model=96, d_state=16, n_layers=2)
for seed in NEW_SEEDS:
    sk = f'seed{seed}'
    if sk in res:
        continue
    t0 = time.time()
    r = R.train_eval(R.NoGateMIMO, kw, Xs, Xa, Y, Xv, Xav, Yv, seed)
    res[sk] = r
    print(f'  seed{seed:<6} MSE={r["mse"]*100:6.2f}  R2={r["r2"]:.4f}  ep={r["best_epoch"]:<4}'
          f'({time.time()-t0:.0f}s)', flush=True)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    os.replace(OUT + '.tmp', OUT)

with open('revision_experiments/results/matrix_results.json', encoding='utf-8') as f:
    M = json.load(f)
old = np.array([M['MIMO-WM-noGate_humanoid_standup'][f'seed{x}']['mse'] for x in OLD_SEEDS]) * 100
new = np.array([res[f'seed{x}']['mse'] for x in NEW_SEEDS if f'seed{x}' in res]) * 100
print('\n=== 结果 ===')
print(f'  原 5 seed  {old.mean():6.2f} ± {old.std():4.2f}   {np.round(old, 2).tolist()}')
if len(new):
    print(f'  新 5 seed  {new.mean():6.2f} ± {new.std():4.2f}   {np.round(new, 2).tolist()}')
    print(f'  两者之差   {new.mean() - old.mean():+.2f}')
    allv = np.concatenate([old, new])
    print(f'  合 10 seed {allv.mean():6.2f} ± {allv.std():4.2f}')
    print('\n  对照（完整模型 MIMO-WM，稿中 53.10）:')
    mm = np.array([M['MIMO-WM_humanoid_standup'][f'seed{x}']['mse'] for x in OLD_SEEDS]) * 100
    print(f'             {mm.mean():6.2f} ± {mm.std():4.2f}')
