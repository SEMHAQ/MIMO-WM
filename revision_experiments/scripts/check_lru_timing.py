"""核对表 1、表 2 中 LRU-WM 时间列（表注 f）所依据的推理延迟。

表注 f 称官方 LRU 的延迟由参考实现的递推写法决定，而非架构计算量。本脚本用与
run_revision_matrix.train_eval 完全相同的计时协议（B=1, T=32, 5 次热身 + 100 次前向,
cuda synchronize）在同一台空闲 GPU 上量出：

  - 本项目各实现（原实现 S4D、w/o 门控、MIMO-WM）：作为参考尺度；
  - 官方 S4D（run_official_s4d.OfficialS4DWorldModel）；
  - 官方 LRU 的两种递推写法（run_official_lru 中的 forward_scan 与 forward_loop）。

若官方 LRU 的两种写法都显著慢于其余各行，即说明该延迟来自 Python 级逐步驱动而非架构。
表注 f 引用的 5.23 ms 即此处 loop 模式的值（scan 为官方默认，即表 1 中的 5.82 ms）。

运行（WSL GPU，工作目录为仓库根）:
  python3 revision_experiments/scripts/check_lru_timing.py
"""
import functools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('revision_experiments/scripts'))

import run_revision_matrix as R
import run_official_s4d as S4D
import run_official_lru as LRU

SD, AD, T = 348, 17, 32
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KW = dict(d_model=96, n_layers=2)


def timeit(model):
    model = model.to(DEV).eval()
    x = torch.randn(1, T, SD, device=DEV)
    a = torch.randn(1, T - 1, AD, device=DEV)
    with torch.no_grad():
        for _ in range(5):
            model(x, a)
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(x, a)
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 100 * 1000


def lru_model(n_state, mode):
    m = LRU.OfficialLRUWorldModel(SD, AD, d_state=n_state, **KW)
    if mode != 'scan':                      # 世界模型包装层不透传 mode, 直接改块的绑定方法
        for blk in m.backbone:
            blk.forward = functools.partial(blk.forward, mode=mode)
    return m


if __name__ == '__main__':
    print('设备:', DEV)
    print('本项目实现(参考尺度):')
    print('  w/o 门控          %.2f ms' % timeit(R.NoGateMIMO(SD, AD, d_state=16, **KW)))
    print('  MIMO-WM           %.2f ms' % timeit(R.MIMOWorldModel(SD, AD, d_state=16, **KW)))
    print('  原实现 S4D        %.2f ms' % timeit(R.SSMWorldModel(SD, AD, d_state=16, **KW)))
    print('官方 S4D:')
    for n in (16, 64):
        print('  N=%-3d             %.2f ms' % (n, timeit(S4D.OfficialS4DWorldModel(SD, AD, d_state=n, **KW))))
    print('官方 LRU:')
    for mode in ('scan', 'loop'):
        for n in (16, 64):
            print('  N=%-3d %-4s        %.2f ms' % (n, mode, timeit(lru_model(n, mode))))
