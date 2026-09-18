#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x86 ONNXRuntime 部署基准：修改稿第 5.7 节 x86 参照延迟的原始结果。

与机载实测（onboard_bench/bench_sbc.py）用同一套 ONNX 文件、同一套输入、
同一计时协议（前 15 次预热，重复次数随窗口递减，取中位数），仅运行平台不同，
因此表 6 与其 x86 参照可直接对照。

用法:  python3 revision_experiments/scripts/bench_deploy_x86_onnx.py
产物:  revision_experiments/results/cpu_onnx_bench.json
"""
import json
import os
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
BENCH = os.path.join(ROOT, 'revision_experiments', 'onboard_bench')
OUT = os.path.join(ROOT, 'revision_experiments', 'results', 'cpu_onnx_bench.json')

WINDOWS = [8, 16, 32, 64]
THREADS = [1, 4]
WARMUP = 15
REPEAT = {8: 200, 16: 150, 32: 100, 64: 50}


def cpu_model():
    try:
        for line in open('/proc/cpuinfo', encoding='utf-8', errors='ignore'):
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def make_sess(path, th):
    so = ort.SessionOptions()
    so.intra_op_num_threads = th
    so.inter_op_num_threads = th
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=so)


def timed(sess, feed, reps):
    for _ in range(WARMUP):
        sess.run(None, feed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {'median_ms': round(ts[len(ts) // 2], 4),
            'p95_ms': round(ts[int(len(ts) * 0.95) - 1], 4),
            'mean_ms': round(statistics.mean(ts), 4), 'reps': reps}


def main():
    ref = np.load(os.path.join(BENCH, 'reference_io.npz'))
    out = {'platform': platform.platform(), 'machine': platform.machine(),
           'processor': cpu_model(), 'cores': os.cpu_count(),
           'python': platform.python_version(), 'onnxruntime': ort.__version__,
           'note': ('x86 ONNXRuntime CPU baseline for Sec. 5.7; same ONNX files, same inputs '
                    'and same timing protocol as the on-board benchmark '
                    '(onboard_bench/bench_sbc.py), platform only differs.'),
           'models': {}}
    print(f"platform: {out['platform']}\ncpu     : {out['processor']} ({out['cores']} cores)"
          f"\nonnxrt  : {ort.__version__}\n")

    for T in WINDOWS:
        base = f'MIMO-WM_T{T}'
        path = os.path.join(BENCH, 'models', base + '.onnx')
        feed = {'states': ref[f'{base}_states'], 'actions': ref[f'{base}_actions']}
        expected = ref[f'{base}_pred']
        rec = {'T': T, 'model_kB': round(os.path.getsize(path) / 1e3, 1), 'latency': {}}
        for th in THREADS:
            sess = make_sess(path, th)
            pred = sess.run(None, feed)[0]
            rec[f'diff_th{th}'] = float(np.max(np.abs(pred - expected)))
            lat = timed(sess, feed, REPEAT[T])
            lat['per_step_ms'] = round(lat['median_ms'] / T, 4)
            rec['latency'][f'th{th}'] = lat
            print(f"{base:<12} th={th}: median {lat['median_ms']:7.4f} ms "
                  f"(per-step {lat['per_step_ms']:6.4f}) | diff {rec[f'diff_th{th}']:.1e}")
        out['models'][base] = rec

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved   :', os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
