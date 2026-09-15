# -*- coding: utf-8 -*-
"""MIMO-WM 初始化变体：让模型从"无门控基线"的位置出发。

动机（可证）：在 MIMOLayer 中
    h ← h + output(DiagSSM(LayerNorm(h))) * sigmoid(gate(·))
令 output = 单位阵、gate ≡ 1，该式逐字退化为 NoGateMIMO 的
    h ← h + DiagSSM(LayerNorm(h))
即 noGate / LRU / 修公平后的 S4D-WM 三者都是 MIMO-WM 的特例，MIMO-WM 的
假设空间严格包含它们。因此 MIMO-WM 在 Standup 上落后只能是优化问题。

本脚本把初始化设到那个"包含关系"的起点上——output 置单位阵、门控偏置置正
（输出门近全开）——使模型在初始时刻恒等于无门控基线，之后只在数据确实需要时
才学出偏离。这是先验的架构选择，与 openGate 修法同源，不涉及任何测试集调参。

口径与 run_revision_matrix.py 完全一致，变体 1/3/5 应分别复现已发表的
53.10 / 52.71 / 50.61（Standup），作为效度自检。

用法（仓库根目录）：
  python3 revision_experiments/scripts/run_init_variants.py --datasets humanoid_standup --seeds 42
  python3 revision_experiments/scripts/run_init_variants.py --datasets humanoid_standup
"""
import sys, os, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

from src.models.mimo_world_model import MIMOWorldModel
import run_revision_matrix as R          # 复用完全相同的训练/评估口径

T = 32
OUT = 'revision_experiments/results/init_variants.json'


class MIMOInitVariant(nn.Module):
    """MIMOWorldModel 的可配置初始化版本，只改初始点，不改结构。"""

    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2,
                 output_identity=False, gate_bias0=None):
        super().__init__()
        self.inner = MIMOWorldModel(state_dim, action_dim,
                                    d_model=d_model, d_state=d_state, n_layers=n_layers)
        with torch.no_grad():
            if output_identity:
                for blk in self.inner.backbone:
                    d = blk.output.weight.shape[0]
                    blk.output.weight.copy_(torch.eye(d))
                    blk.output.bias.zero_()
            if gate_bias0 is not None:
                for blk in self.inner.backbone:
                    blk.gate.bias.fill_(gate_bias0)

    def forward(self, states, actions, mode='conv'):
        return self.inner(states, actions, mode=mode)


def t(kw):
    return lambda sd, ad: dict(kw, state_dim=sd, action_dim=ad)


VARIANTS = {
    'MIMO-WM(原样)':            (MIMOInitVariant, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    'MIMO-WM(output=I)':        (MIMOInitVariant, t({'d_model': 96, 'd_state': 16, 'n_layers': 2,
                                                     'output_identity': True})),
    'MIMO-WM(门控开)':          (MIMOInitVariant, t({'d_model': 96, 'd_state': 16, 'n_layers': 2,
                                                     'gate_bias0': 4.0})),
    'MIMO-WM(output=I+门控开)': (MIMOInitVariant, t({'d_model': 96, 'd_state': 16, 'n_layers': 2,
                                                     'output_identity': True, 'gate_bias0': 4.0})),
}
CONTROLS = {'w/o门控(控制组)': (R.NoGateMIMO, t({'d_model': 96, 'd_state': 16, 'n_layers': 2}))}


if __name__ == '__main__':
    argv = sys.argv[1:]

    def multi(flag, cast=str):
        """取 flag 后的连续非 flag 参数。原写法的 argv[i+1:][:2] 会把下一个
        flag 也吃进来（--datasets X --seeds Y -> dss=['X','--seeds']，
        跑到第二个"数据集"时 KeyError）。"""
        if flag not in argv:
            return []
        out = []
        for a in argv[argv.index(flag) + 1:]:
            if a.startswith('--'):
                break
            out.append(cast(a))
        return out

    seeds = multi('--seeds', int)[:5] or R.SEEDS
    dss = multi('--datasets')[:2] or ['humanoid_standup']
    only = multi('--variants')          # 可选：只跑指定变体（按名字精确匹配）
    ALL = list(VARIANTS.items()) + list(CONTROLS.items())

    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    for ds in dss:
        cfg = R.DATASETS[ds]
        print('\n加载 %s ...' % ds, flush=True)
        eps_tr = R.load_eps(cfg['dir'], 'train')
        eps_vl = R.load_eps(cfg['dir'], 'val')
        a = np.concatenate([s for s, _ in eps_tr])
        m, s = a.mean(0), a.std(0)
        Xs, Xa, Y = R.make_data(eps_tr, m, s, T)
        Xv, Xav, Yv = R.make_data(eps_vl, m, s, T)

        for name, (ModelClass, kwfn) in ALL if not only else [(n, v) for n, v in ALL if n in only]:
            key = '%s_%s' % (name, ds)
            res.setdefault(key, {})
            for seed in seeds:
                sk = 'seed%d' % seed
                if sk in res[key]:
                    print('  %s %s: 已有, 跳过' % (key, sk), flush=True)
                    continue
                r = R.train_eval(ModelClass, kwfn(cfg['sd'], cfg['ad']),
                                 Xs, Xa, Y, Xv, Xav, Yv, seed)
                res[key][sk] = r
                print('  %-30s %s  MSE=%.2f  R2=%.3f  ep=%d'
                      % (name, sk, r['mse'] * 100, r['r2'], r['best_epoch']), flush=True)
                json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('\n=== 汇总 (%s) ===' % ', '.join(dss), flush=True)
    for ds in dss:
        print('[%s]' % ds)
        for name in [n for n, _ in ALL if not only or n in only]:
            v = res.get('%s_%s' % (name, ds), {})
            ms = [x['mse'] * 100 for x in v.values()]
            if ms:
                print('  %-30s %.2f ± %.2f  (n=%d)' % (name, np.mean(ms), np.std(ms), len(ms)))
