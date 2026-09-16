# -*- coding: utf-8 -*-
"""官方参考 S4D 基线（表 1、表 2 中 S4D-WM 一行即取自本脚本）。

为什么改用参考实现：
  run_revision_matrix.py 的 'S4D-WM' 绑的是 src/models/ssm_world_model.py 的
  SSMWorldModel，它相对本项目的 NoGateMIMO（MIMO-WM 去掉门控）另有两处设置：
    (a) decoder 之前的一个 LayerNorm；
    (b) 自定义 Xavier + 零偏置初始化。
  即该实现是本项目自身骨架的一个变体，而非 S4D 的作者实现。其结果仍保留在
  matrix_results.json 中（键名 'S4D-WM'）供核对，但不作为表 1、表 2 中
  'S4D-WM' 一行的出处。本脚本改用 S4D 论文作者所属实验室发布的配套实现，
  按表内各模型的共用骨架外围绕接后重跑。

代码来源（逐行照抄，仅两处机械改动，见下）：
    https://github.com/HazyResearch/state-spaces
    -- models/s4/s4d.py  的 S4DKernel 类与 S4D 类
    该文件首行自述："Minimal version of S4D with extra options and features
    stripped out, for pedagogical purposes."，是 S4D 论文（Gu, Gupta, Goel, R\'e,
    "On the Parameterization and Initialization of Diagonal State Space Models",
    NeurIPS 2022）作者所属实验室发布的配套最小实现，对应论文中的 S4D-Lin
    （log_A_real = log 0.5 <=> A_real = -0.5；A_imag = pi*n）。

机械改动（仅此两处，均可逐行核对）：
  1. einops.repeat(arange(N//2), 'n -> h n', h=H)
       -> arange(N//2).unsqueeze(0).repeat(H, 1)
     （仅去掉 einops 依赖；repeat 会物化为新的连续张量，与 einops 语义一致，
       且避免 expand 的零步长视图进入 nn.Parameter。）
  2. 删去 from src.models.nn import DropoutNd 及其分支。
     参考实现中该分支仅当 dropout > 0 时启用，本脚本取 dropout=0.0，
     参考实现自身在该取值下也解析为 nn.Identity()，故行为不变。

外围绕接（非 source-spaces 代码，据实标注）：
  最小参考 s4d.py 只含【层本身】——其 S4D.forward 已内含 FFT 卷积、D 跳连、
  GELU、GLU 输出变换与 dropout，但没有外围的前置归一化与残差。S4D 系列的标准用法
  是 pre-norm 残差块（见同仓库 src/models/sequence/modules/s4block.py；该文件与
  仓库的 registry / FFTConv / SequenceModule 深度耦合，不自包含，故未直接搬运）。
  本脚本因此把官方 S4D 层嵌入 pre-norm 残差块：
        x = x + S4D(LayerNorm(x))
  这与本文件对 LRU 基线（run_official_lru.py 用官方 DWNBlock：
  LayerNorm -> LRU -> GLU -> Dropout -> 残差）的处理口径一致——两者都是
  "pre-norm -> SSM 层 -> GLU -> 残差"，差别只在 SSM 层本身。
  外围 encoder / decoder 沿用本项目其余模型的共用骨架，以便与表内各模型可比。

用法（仓库根目录）：
  python revision_experiments/scripts/run_official_s4d.py --datasets humanoid --seeds 42
  python revision_experiments/scripts/run_official_s4d.py --datasets humanoid humanoid_standup \
      --seeds 42 123 456 789 1024
"""
import sys, os, json, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

import run_revision_matrix as R          # 复用完全相同的训练/评估口径

T = 32
OUT = 'revision_experiments/results/official_s4d.json'


# =====================================================================
# 以下两个类：models/s4/s4d.py 逐行照抄（仅上述两处机械改动，已就地标注）
# =====================================================================
class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        # 机械改动 1：原为 einops.repeat(torch.arange(N//2), 'n -> h n', h=H)
        A_imag = math.pi * torch.arange(N//2).unsqueeze(0).repeat(H, 1)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # 机械改动 2：原为 dropout_fn = DropoutNd；dropout=0 时参考实现亦解析为
        # nn.Identity()，故此处直接用 nn.Identity()，行为不变。
        self.dropout = nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified


# =====================================================================
# 以下：本项目包装（非 source-spaces 代码）
# =====================================================================
class OfficialS4DBlock(nn.Module):
    """pre-norm 残差块：x = x + S4D(LayerNorm(x))。

    与 run_official_lru.py 的官方 DWNBlock（LayerNorm -> LRU -> GLU -> Dropout -> 残差）
    同构；S4D.forward 内部已含 GELU 与 GLU 输出变换，故此处不再另接 FFN。
    """

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.s4d = S4D(d_model, d_state=d_state, dropout=0.0, transposed=False)

    def forward(self, x):
        y, _ = self.s4d(self.ln(x))
        return x + y


class OfficialS4DWorldModel(nn.Module):
    """共用骨架 + 官方 S4D 块。"""

    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model),
                                     nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = nn.ModuleList([OfficialS4DBlock(d_model, d_state=d_state)
                                       for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model),
                                     nn.GELU(), nn.Linear(d_model, state_dim))

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            actions = torch.cat([torch.zeros(states.shape[0], states.shape[1] - actions.shape[1],
                                             actions.shape[-1], device=actions.device), actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        h = self.encoder(x)
        for blk in self.backbone:
            h = blk(h)
        return states[:, -1] + self.decoder(h[:, -1])


def t(kw):
    return lambda sd, ad: dict(kw, state_dim=sd, action_dim=ad)


VARIANTS = {
    # d_state=16：与表内其余模型的 d_state 一致，参数量可比
    'S4D-WM(官方, N=16)': (OfficialS4DWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    # d_state=64：官方 S4DKernel 的默认值，作为敏感性对照
    'S4D-WM(官方, N=64)': (OfficialS4DWorldModel, t({'d_model': 96, 'd_state': 64, 'n_layers': 2})),
}
# 控制组：效度自检，应逐位重现已发表值（S4D 已发表 31.18 / 51.90；noGate 20.55 / 50.61）
CONTROLS = {
    'S4D-WM(原实现)':  (R.SSMWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    'w/o门控(控制组)': (R.NoGateMIMO,    t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    'MIMO-WM(完整)':   (R.MIMOWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
}


if __name__ == '__main__':
    argv = sys.argv[1:]

    def multi(flag, cast=str):
        """取 flag 后的连续非 flag 参数（勿用 argv[i+1:][:2]，会吃进下一个 flag）。"""
        if flag not in argv:
            return []
        out = []
        for a in argv[argv.index(flag) + 1:]:
            if a.startswith('--'):
                break
            out.append(cast(a))
        return out

    seeds = multi('--seeds', int)[:5] or [42]
    dss = multi('--datasets')[:2] or ['humanoid']
    only = multi('--variants')
    pick = lambda d: {k: v for k, v in d.items() if not only or any(o in k for o in only)}
    reg, ctrl = pick(dict(VARIANTS)), pick(dict(CONTROLS))

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

        for name, (ModelClass, kwfn) in list(reg.items()) + list(ctrl.items()):
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
                print('  %-24s %s  MSE=%.2f  R2=%.3f  params=%.3fM  ep=%d'
                      % (name, sk, r['mse'] * 100, r['r2'], r['params_m'], r['best_epoch']),
                      flush=True)
                # 显式 utf-8：Windows 下 open(...,'w') 默认 cp936，会写出 GBK 文件
                with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                os.replace(OUT + '.tmp', OUT)

    print('\n=== 汇总 (%s) ===' % ', '.join(dss), flush=True)
    for ds in dss:
        print('[%s]' % ds)
        for name in list(reg) + list(CONTROLS):
            v = res.get('%s_%s' % (name, ds), {})
            ms = [x['mse'] * 100 for x in v.values()]
            if ms:
                print('  %-24s %.2f ± %.2f  (n=%d)'
                      % (name, np.mean(ms), np.std(ms), len(ms)))
