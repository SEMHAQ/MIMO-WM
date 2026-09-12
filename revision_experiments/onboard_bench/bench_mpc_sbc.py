#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机载 CEM-MPC 规划耗时实测（在世界模型上用 ONNXRuntime 批量评估候选序列）。

依赖: pip install onnxruntime numpy
用法: python3 bench_mpc_sbc.py
产物: bench_result_mpc.json（连同终端输出发回）

说明: 代价函数与论文一致 J = Σ_h [‖ŝ_h − s_ref‖²_Q + ‖a_h‖²_R] (Q=1, R=0.01);
      每个控制时刻重新规划; 每个候选序列用世界模型前向展开 H 步, K 个候选批量评估.
"""
import json, os, platform, statistics, time
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'models', 'MIMO-WM_T8_batch.onnx')
SD, AD, T = 348, 17, 8
Q, R = 1.0, 0.01
N_STEPS = 3          # 测 3 个控制时刻
THREADS = 4

CONFIGS = [
    {'name': '论文配置 K256-M5-H10', 'K': 256, 'Ne': 32, 'M': 5, 'H': 10},
    {'name': '缩减配置 K32-M3-H5',   'K': 32,  'Ne': 8,  'M': 3, 'H': 5},
]


def make_sess(path, th):
    so = ort.SessionOptions()
    so.intra_op_num_threads = th
    so.inter_op_num_threads = th
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=so)


def plan_once(sess, s0, a0, target, cfg, rng):
    """执行一次 CEM 规划, 返回 (耗时秒, 迭代轮数)."""
    K, Ne, M, H = cfg['K'], cfg['Ne'], cfg['M'], cfg['H']
    mu = np.zeros((H, AD), dtype=np.float32)
    sigma = np.ones((H, AD), dtype=np.float32)
    S = np.repeat(s0[None], K, axis=0)          # (K,T,SD)
    A = np.repeat(a0[None], K, axis=0)          # (K,T-1,AD)
    t0 = time.perf_counter()
    for _ in range(M):
        cand = (mu[None] + sigma[None] * rng.standard_normal((K, H, AD))).astype(np.float32)
        cand = np.clip(cand, -1, 1)
        cost = np.zeros(K, dtype=np.float32)
        S = np.repeat(s0[None], K, axis=0).copy()
        A = np.repeat(a0[None], K, axis=0).copy()
        for h in range(H):
            A[:, -1, :] = cand[:, h, :]
            pred = sess.run(None, {'states': S, 'actions': A})[0]      # (K,SD)
            cost += Q * np.sum((pred - target[None]) ** 2, axis=1) + R * np.sum(cand[:, h, :] ** 2, axis=1)
            S = np.concatenate([S[:, 1:, :], pred[:, None, :]], axis=1)
            A = np.concatenate([A[:, 1:, :], np.zeros((K, 1, AD), dtype=np.float32)], axis=1)
        idx = np.argsort(cost)[:Ne]
        elite = cand[idx]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-6
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(0)
    out = {'platform': platform.platform(), 'machine': platform.machine(),
           'processor': platform.processor(), 'onnxruntime': ort.__version__,
           'threads': THREADS, 'model': os.path.basename(MODEL), 'configs': {}}
    print(f"platform : {out['platform']}")
    print(f"onnxrt   : {ort.__version__} | threads {THREADS} | model {os.path.basename(MODEL)}\n")
    sess = make_sess(MODEL, THREADS)
    s0 = rng.standard_normal((T, SD)).astype(np.float32)
    a0 = rng.standard_normal((T - 1, AD)).astype(np.float32)

    for cfg in CONFIGS:
        target = rng.standard_normal(SD).astype(np.float32)
        steps = N_STEPS
        ts = []
        for i in range(steps):
            dt = plan_once(sess, s0, a0, target, cfg, rng)
            ts.append(dt)
            print(f"{cfg['name']:<24} 控制步 {i+1}: {dt*1e3:9.1f} ms  ({1/dt:7.3f} Hz)")
            if i == 0 and dt > 60:      # 全量配置过慢时只测一步
                print('  (单步超过 60 s, 仅测 1 步)')
                break
        med = statistics.median(ts)
        out['configs'][cfg['name']] = {'K': cfg['K'], 'Ne': cfg['Ne'], 'M': cfg['M'], 'H': cfg['H'],
                                       'median_ms': round(med * 1e3, 1), 'hz': round(1 / med, 3),
                                       'steps': len(ts)}
        print()
    with open(os.path.join(HERE, 'bench_result_mpc.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved : bench_result_mpc.json（请连同终端输出一起发回）')


if __name__ == '__main__':
    main()
