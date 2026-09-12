# -*- coding: utf-8 -*-
"""导出边缘部署用的 ONNX 模型集合 + 参考输入/输出（在 WSL 下运行，需要 torch + onnxruntime）。"""
import os, sys
import numpy as np
import torch
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from revision_experiments.scripts.deploy_mimo import build          # MIMO 部署核
from src.models.baselines import TransformerWorldModel, LSTMWorldModel

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

def main():
    _, dep_mimo = build(0)
    dep_mimo.eval()
    trans = TransformerWorldModel(SD, AD, d_model=96, nhead=4, n_layers=2).eval()
    lstm = LSTMWorldModel(SD, AD, hidden_dim=96, n_layers=2).eval()

    jobs = [(dep_mimo, 'MIMO-WM', T) for T in (8, 16, 32, 64)]
    jobs += [(trans, 'Transformer-WM', T) for T in (8, 32)]
    jobs += [(lstm, 'LSTM-WM', T) for T in (8, 32)]

    ref = {}
    for model, name, T in jobs:
        path, st, ac = export(model, name, T)
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        pred = sess.run(None, {'states': st.numpy(), 'actions': ac.numpy()})[0]
        ref[f'{name}_T{T}_states'] = st.numpy()
        ref[f'{name}_T{T}_actions'] = ac.numpy()
        ref[f'{name}_T{T}_pred'] = pred
        print(f'{name}_T{T}: {os.path.getsize(path)/1e3:6.1f} kB  pred {pred.shape}')

    np.savez_compressed(os.path.join(HERE, 'reference_io.npz'), **ref)
    print('reference_io.npz saved;', len(jobs), 'models')

if __name__ == '__main__':
    main()
