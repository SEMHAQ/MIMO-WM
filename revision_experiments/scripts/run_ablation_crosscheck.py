# -*- coding: utf-8 -*-
"""交叉验证：消融表的 w/o门控(20.36) 与表1 的 w/o 门控(20.55) 为何不同。

读代码得到的假设（未经验证，故有此脚本）：两处无门控的结构逐行等价——
  run_ablation_mimo.py 的 MIMOLayer(use_glu=False)  =>  x + DiagSSM(LayerNorm(x))
  run_revision_matrix.py 的 NoGateMIMO              =>  h + DiagSSM(LayerNorm(h))
——数据管线（训练集统计量做 z-score、同一 make_data）、优化器、调度、早停、
"报最终 epoch 而非 best"的汇报口径也全部一致。唯一差异是隐维度：
消融用 d_model=128（参数量 0.142M），表1 用 d_model=96（0.101M）。

本脚本用表1 的 harness（R.train_eval）跑 NoGateMIMO 的两个 d_model：
  若 128 -> 20.36 且 96 -> 20.55 => 差异全部由 d_model 解释，假设成立；
  否则说明两脚本之间还有别的差异，需要继续查（则本脚本的输出即为反例）。

96 这一档同时充当效度自检：应逐位复现已发表的 20.55 ± 0.05。

用法（仓库根目录）：
  python3 revision_experiments/scripts/run_ablation_crosscheck.py
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

import run_revision_matrix as R

T = 32
OUT = 'revision_experiments/results/ablation_crosscheck.json'

CONFIGS = {
    'noGate_d96(复现表1)':   {'d_model': 96,  'd_state': 16, 'n_layers': 2},
    'noGate_d128(复现消融)': {'d_model': 128, 'd_state': 16, 'n_layers': 2},
}


if __name__ == '__main__':
    argv = sys.argv[1:]

    def multi(flag, cast=str):
        if flag not in argv:
            return []
        out = []
        for a in argv[argv.index(flag) + 1:]:
            if a.startswith('--'):
                break
            out.append(cast(a))
        return out

    seeds = multi('--seeds', int)[:5] or [42, 123, 456, 789, 1024]
    ds = (multi('--datasets')[:1] or ['humanoid'])[0]

    cfg = R.DATASETS[ds]
    print('\n加载 %s ...' % ds, flush=True)
    eps_tr = R.load_eps(cfg['dir'], 'train')
    eps_vl = R.load_eps(cfg['dir'], 'val')
    a = np.concatenate([s for s, _ in eps_tr])
    m, s = a.mean(0), a.std(0)
    Xs, Xa, Y = R.make_data(eps_tr, m, s, T)
    Xv, Xav, Yv = R.make_data(eps_vl, m, s, T)

    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    for name, kw in CONFIGS.items():
        key = '%s_%s' % (name, ds)
        res.setdefault(key, {})
        for seed in seeds:
            sk = 'seed%d' % seed
            if sk in res[key]:
                print('  %s %s: 已有, 跳过' % (key, sk), flush=True)
                continue
            r = R.train_eval(R.NoGateMIMO, dict(kw, state_dim=cfg['sd'], action_dim=cfg['ad']),
                             Xs, Xa, Y, Xv, Xav, Yv, seed)
            res[key][sk] = r
            print('  %-24s %s  MSE=%.2f  R2=%.3f  P=%.3fM  ep=%d'
                  % (name, sk, r['mse'] * 100, r['r2'], r['params_m'], r['best_epoch']), flush=True)
            json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('\n=== 汇总 (%s) ===' % ds, flush=True)
    for name in CONFIGS:
        v = res.get('%s_%s' % (name, ds), {})
        ms = [x['mse'] * 100 for x in v.values()]
        if ms:
            print('  %-24s %.2f ± %.2f  (n=%d)'
                  % (name, np.mean(ms), np.std(ms), len(ms)))
    print('\n对照：表1 w/o门控(d96)=20.55  消融 w/o门控(d128)=20.36', flush=True)
