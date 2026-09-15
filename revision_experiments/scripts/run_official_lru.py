# -*- coding: utf-8 -*-
"""官方参考 LRU 基线（用于替换此前两版，均为非忠实实现）。

为什么重写两次：
  * 第 1 版（run_revision_matrix.py 的 _LRUBlk，即已发表的 "LRU-WM"）：
    gate 是 nn.Parameter(d_model)，一个【与输入无关】的静态逐通道常数。
    LRU 的定义性特征是输入依赖门控，故那份实现不是 LRU。
  * 第 2 版（run_true_lru.py 的 TrueLRULayer）：参数化对了，但结构有两处错：
    (a) GLU 挂在了递归【之前】（LayerNorm → GLU → LRU → out_proj），
        官方 DWNBlock 是 LayerNorm → LRU → GLU → Dropout → 残差；
    (b) B/C 写成实数逐通道对角配对、D 写成向量；官方 B (N,H)、C (H,N) 全为
        【复数】，D 为完整 (H,H) 实矩阵，混合是全 H×H 的：
            x_k = A x_{k-1} + B u_k
            y_k = Re[C x_k] + D u_k
本版即为此前两版的更正。

代码来源（逐行照抄，仅两处机械改动，见下）：
    https://github.com/forgi86/sysid-pytorch-lru
    -- lru/linear.py        的 LRU 类
    -- lru/scan_utils.py    的 safe_map / combine / _scan / associative_scan /
                             _interleave / binary_operator_diag
    -- lru/architectures.py 的 DWNConfig / MLP / GLU / DWNBlock

出处性质（据实说明，勿误引）：这是社区维护的 PyTorch 实现，README 自称
"A PyTorch implementation of DeepMind's LRU (arXiv:2303.06349)"，**不是 DeepMind
的官方发布**。选它的理由是逐条对应论文方程（对角复 A、满复 B/C、满实 D）且是
最广泛被引用的 PyTorch 参考版。若审稿要求官方 artifact，此处应整体替换。

机械改动（仅此两处，均可逐行核对）：
  1. LRU.forward 的 match 语句 → if/elif（本机 Python 3.9.7，match 需 3.10+）。
  2. 删去 linear.py 末尾的 __main__ 演示段。
另：@torch.compiler.disable 做了兼容包装（torch 2.8 有该属性，包装是恒等的）。

与其余 6 个模型的口径衔接：官方 DWN 骨架是 Linear(n_u→d) → blocks → Linear(d→n_y)。
本项目其余基线与 MIMO-WM 共用 "两层 encoder / 两层 decoder + 末步残差" 的骨架，
为可比起见本脚本【保留该共用骨架，只把递归块换成官方 DWNBlock】。这是刻意的
口径选择，改动的是块本身而非外围。

用法（仓库根目录）：
  python revision_experiments/scripts/run_official_lru.py --datasets humanoid --seeds 42
  python revision_experiments/scripts/run_official_lru.py --datasets humanoid humanoid_standup
"""
import sys, os, json, math
from dataclasses import dataclass
from typing import overload, Callable, Iterable, List, TypeVar, Any, Tuple
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

import run_revision_matrix as R          # 复用完全相同的训练/评估口径

T = 32
OUT = 'revision_experiments/results/official_lru.json'

# @torch.compiler.disable 的兼容包装：torch 2.8 有该属性，包装恒等
_if_dis = getattr(getattr(torch, 'compiler', None), 'disable', None)
compiler_disable = _if_dis if callable(_if_dis) else (lambda f: f)


# =====================================================================
# 以下三段：lru/scan_utils.py 逐行照抄
# =====================================================================
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], Any], __arg1: Iterable[T1]) -> List[Any]: ...


@overload
def safe_map(f: Callable[[T1, T2], Any], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> List[Any]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))


def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(tree, operator,
                            [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
                            [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(tree, operator,
                             [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])
    else:
        even_elems = combine(tree, operator,
                             odd_elems,
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0 else
        result if result.shape.numel() > 0 else
        torch.ops.aten.slice(elem, axis, 0, 1)
        for (elem, result) in zip(elems, even_elems)]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)


def _interleave(a, b, axis):
    if b_trunc := (a.shape[axis] == b.shape[axis] + 1):
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        interleaved = torch.ops.aten.slice(interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1)
    return interleaved


@torch.jit.script
def binary_operator_diag(q_i: Tuple[torch.Tensor, torch.Tensor],
                         q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


# =====================================================================
# 以下：lru/linear.py 的 LRU 类，逐行照抄（仅 match → if/elif）
# =====================================================================
class LRU(nn.Module):
    def __init__(
        self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))          # N, U
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))          # H, N

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        lambda_re = lambda_abs * torch.cos(lambda_phase)
        lambda_im = lambda_abs * torch.sin(lambda_phase)
        lambdas = torch.complex(lambda_re, lambda_im)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def forward_loop(self, input, state=None):
        # Input size: (B, L, H)
        lambdas, B, C, D = self.ss_params()
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )
        states = []
        for u_step in input.split(1, dim=1):        # 1 is the time dimension
            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)
        states = torch.stack(states, 1)
        output = (states @ C.mT).real + input @ D.T
        return output

    @compiler_disable
    def forward_scan(self, input, state=None):
        # Only handles input of size (B, L, H)
        lambdas, B, C, D = self.ss_params()
        lambda_elements = lambdas.tile(input.shape[1], 1)
        Bu_elements = input.to(B.dtype) @ B.T
        if state is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + lambdas * state
        inner_state_fn = lambda Bu_seq: associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]
        inner_states = torch.vmap(inner_state_fn)(Bu_elements)
        y = (inner_states @ C.T).real + input @ D.T
        return y

    def forward(self, input, state=None, mode="scan"):
        if state is None:
            state = torch.view_as_complex(
                torch.zeros((self.state_features, 2), device=input.device)
            )
        # 原文为 match mode: case "scan" / case "loop"（Python 3.10+），此处等价改写
        if mode == "scan":
            y = self.forward_scan(input, state)
        elif mode == "loop":
            y = self.forward_loop(input, state)
        else:
            raise ValueError('mode must be "scan" or "loop", got %r' % (mode,))
        return y


# =====================================================================
# 以下：lru/architectures.py 的 DWNConfig / MLP / GLU / DWNBlock，逐行照抄
# =====================================================================
@dataclass
class DWNConfig:
    d_model: int = 10
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    bias: bool = True
    rmin: float = 0.0
    rmax: float = 1.0
    max_phase: float = 2 * math.pi
    ff: str = "GLU"


class MLP(nn.Module):
    """ Standard Transformer MLP """
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """ The static nonlinearity used in the S4 paper"""
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x


class DWNBlock(nn.Module):
    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lru = LRU(config.d_model, config.d_model, config.d_state,
                       rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        if config.ff == "GLU":
            self.ff = GLU(config)
        elif config.ff == "MLP":
            self.ff = MLP(config)
        else:
            raise ValueError('ff must be "GLU" or "MLP", got %r' % (config.ff,))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None, mode="scan"):
        z = x
        z = self.ln(z)              # prenorm
        z = self.lru(z, state, mode)
        z = self.ff(z)              # MLP or GLU
        z = self.dropout(z)
        x = z + x                   # Residual connection
        return x


# =====================================================================
# 以下：本项目包装（非官方代码）——沿用其余 6 个模型的共用 encoder/decoder 骨架
# =====================================================================
class OfficialLRUWorldModel(nn.Module):
    """共用骨架 + 官方 DWNBlock 作为递归块。"""

    def __init__(self, state_dim, action_dim, d_model=96, d_state=16, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim + action_dim, d_model),
                                     nn.GELU(), nn.Linear(d_model, d_model))
        cfg = DWNConfig(d_model=d_model, d_state=d_state, n_layers=n_layers)
        self.backbone = nn.ModuleList([DWNBlock(cfg) for _ in range(n_layers)])
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
    'LRU-WM(官方, N=16)': (OfficialLRUWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    # d_state=64：官方 DWNConfig 的默认值，作为敏感性对照
    'LRU-WM(官方, N=64)': (OfficialLRUWorldModel, t({'d_model': 96, 'd_state': 64, 'n_layers': 2})),
}
# 控制组：效度自检，应逐位重现已发表值（Standup 50.35 / Humanoid 20.42；noGate 50.61 / 20.55）
CONTROLS = {
    'LRU-WM(原实现)':  (R.LRUWorldModel, t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
    'w/o门控(控制组)': (R.NoGateMIMO,    t({'d_model': 96, 'd_state': 16, 'n_layers': 2})),
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
