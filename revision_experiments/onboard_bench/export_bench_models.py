# -*- coding: utf-8 -*-
"""导出边缘部署用的 ONNX 模型集合 + 参考输入/输出（在 WSL 下运行，需要 torch + onnxruntime）。

导出的模型分两类：
  主模型  MIMO-WM（T=8/16/32/64）—— 单步递推部署核，纯实数、全标准算子；
  对照    MIMO-WM-noGate / Transformer-WM / Transformer-Reg / LSTM-WM / GRU-WM / TCN-WM
          （T=8/32）—— 用于在**同一块嵌入式硬件上**对照推理耗时与内存，
          以支撑"轻量化硬件表现"的结论（审稿意见 2.5）。

权重为随机初始化：延迟与内存不依赖权重取值，且参考输出在同一进程内生成，
板上脚本以 max abs diff 与参考输出比对，可验证板上推理与 x86 逐元素一致。

导出清单同时写入 models/manifest.json，供表 6 表注与 README 引用。
"""
import json
import os
import sys

import numpy as np
import torch
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

from revision_experiments.scripts.deploy_mimo import build          # MIMO 部署核
from revision_experiments.scripts.run_revision_matrix import NoGateMIMO
from src.models.baselines import (TransformerWorldModel, LSTMWorldModel,
                                  GRUWorldModel, TCNWorldModel)

SD, AD = 348, 17
MODELS_DIR = os.path.join(HERE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

torch.manual_seed(0)


def export(model, name, T):
    st = torch.randn(1, T, SD)
    ac = torch.randn(1, T - 1, AD)
    path = os.path.join(MODELS_DIR, f'{name}_T{T}.onnx')
    torch.onnx.export(model, (st, ac), path, input_names=['states', 'actions'],
                      output_names=['pred'], opset_version=17)
    return path, st, ac


def jobs():
    _, dep_mimo = build(0)
    dep_mimo.eval()
    out = [(dep_mimo, 'MIMO-WM', T) for T in (8, 16, 32, 64)]

    # 对照模型：T=8 与 32 覆盖"短窗实时"与"论文主口径"两种设置
    baselines = [
        ('MIMO-WM-noGate', NoGateMIMO(SD, AD, d_model=96, d_state=16, n_layers=2)),
        ('Transformer-WM', TransformerWorldModel(SD, AD, d_model=96, nhead=4, n_layers=2)),
        ('Transformer-Reg', TransformerWorldModel(SD, AD, d_model=192, nhead=6, n_layers=3)),
        ('LSTM-WM', LSTMWorldModel(SD, AD, hidden_dim=96, n_layers=2)),
        ('GRU-WM', GRUWorldModel(SD, AD, hidden_dim=96, n_layers=2)),
        ('TCN-WM', TCNWorldModel(SD, AD, d_model=96, n_layers=2)),
    ]
    for name, m in baselines:
        out += [(m.eval(), name, T) for T in (8, 32)]
    return out


def main():
    ref, manifest = {}, {}
    for model, name, T in jobs():
        try:
            path, st, ac = export(model, name, T)
            sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            pred = sess.run(None, {'states': st.numpy(), 'actions': ac.numpy()})[0]
        except Exception as e:                      # 某个模型导不出来不应中断其余模型
            print(f'{name}_T{T}: 导出失败, 跳过 ({type(e).__name__}: {e})')
            continue
        ref[f'{name}_T{T}_states'] = st.numpy()
        ref[f'{name}_T{T}_actions'] = ac.numpy()
        ref[f'{name}_T{T}_pred'] = pred
        kb = os.path.getsize(path) / 1e3
        manifest[f'{name}_T{T}'] = {
            'T': T,
            'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
            'onnx_kB': round(kb, 1),
        }
        print(f'{name}_T{T}: {kb:6.1f} kB  pred {pred.shape}')

    np.savez_compressed(os.path.join(HERE, 'reference_io.npz'), **ref)
    with open(os.path.join(MODELS_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'reference_io.npz saved; {len(manifest)} models')


if __name__ == '__main__':
    main()
