# -*- coding: utf-8 -*-
"""导出支持动态批量的 MIMO-WM ONNX（供机载 CEM-MPC 规划测速用）。"""
import os, sys
import numpy as np
import torch
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from revision_experiments.scripts.deploy_mimo import build

SD, AD, T = 348, 17, 8
MODELS = os.path.join(HERE, 'models')
os.makedirs(MODELS, exist_ok=True)

_, dep = build(0)
dep.eval()
st = torch.randn(1, T, SD)
ac = torch.randn(1, T - 1, AD)
path = os.path.join(MODELS, 'MIMO-WM_T8_batch.onnx')
torch.onnx.export(dep, (st, ac), path,
                  input_names=['states', 'actions'], output_names=['pred'],
                  dynamic_axes={'states': {0: 'batch'}, 'actions': {0: 'batch'}},
                  opset_version=17)

# 自检：批量推理可用
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
y1 = sess.run(None, {'states': st.numpy(), 'actions': ac.numpy()})[0]
K = 16
stK = np.repeat(st.numpy(), K, axis=0)
acK = np.repeat(ac.numpy(), K, axis=0)
yK = sess.run(None, {'states': stK, 'actions': acK})[0]
print('exported', path, round(os.path.getsize(path)/1e3, 1), 'kB')
print('batch ok:', yK.shape, '| max diff vs single:', float(np.max(np.abs(yK - np.repeat(y1, K, axis=0)))))
