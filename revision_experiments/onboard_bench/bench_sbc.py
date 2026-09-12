#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIMO-WM 机载平台部署测速（树莓派/香橙派/Jetson 等 ARM 设备）。

依赖:  pip install onnxruntime numpy
用法:  python3 bench_sbc.py            # 全部模型
       python3 bench_sbc.py MIMO-WM    # 只测某个模型
产物:  bench_result.json（连同终端输出一起发回）

测量内容:
  - 正确性: 与 reference_io.npz 参考输出比对 (max abs diff)
  - 延迟:  每个模型在 T=8/16/32/64（视模型而定）与线程 1/4 下的 mean/median/p95,
           并给出整窗延迟与折算单步延迟
  - 内存:  模型权重体积 + 进程峰值常驻内存 (VmHWM)
  - 温度:  运行前后 CPU 温度（若有 sysfs 传感器），用于排除降频影响
  - 环境:  平台 / CPU / 核数 / onnxruntime 版本
"""
import glob, json, os, platform, re, statistics, sys, time
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models')
THREADS = [1, 4]
WARMUP = 15
REPEAT = {8: 200, 16: 150, 32: 100, 64: 50}


def read_temp():
    vals = []
    for p in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
        try:
            vals.append(round(int(open(p).read().strip()) / 1000.0, 1))
        except Exception:
            pass
    return max(vals) if vals else None


def cpu_model():
    try:
        for line in open('/proc/cpuinfo', encoding='utf-8', errors='ignore'):
            low = line.lower()
            if low.startswith('model name') or low.startswith('hardware') or low.startswith('model\t'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def vm_hwm_mb():
    try:
        for line in open('/proc/self/status', encoding='utf-8'):
            if line.startswith('VmHWM'):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return None


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
    return {'mean_ms': round(statistics.mean(ts), 4), 'median_ms': round(ts[len(ts) // 2], 4),
            'p95_ms': round(ts[int(len(ts) * 0.95) - 1], 4), 'reps': reps}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ref = np.load(os.path.join(HERE, 'reference_io.npz'))
    out = {'platform': platform.platform(), 'machine': platform.machine(),
           'processor': cpu_model(), 'cores': os.cpu_count(),
           'python': platform.python_version(), 'onnxruntime': ort.__version__,
           'temp_before_C': read_temp(), 'models': {}}
    print(f"platform : {out['platform']}")
    print(f"cpu      : {out['processor']} ({out['machine']}, {out['cores']} cores)")
    print(f"onnxrt   : {ort.__version__} | temp {out['temp_before_C']} C\n")

    for path in sorted(glob.glob(os.path.join(MODELS, '*.onnx'))):
        base = os.path.basename(path)[:-5]
        m = re.match(r'(.+)_T(\d+)$', base)
        if not m:
            continue
        name, T = m.group(1), int(m.group(2))
        if only and name != only:
            continue
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
            print(f"{base:<20} th={th}: median {lat['median_ms']:7.3f} ms "
                  f"(per-step {lat['per_step_ms']:6.4f}) | diff {rec[f'diff_th{th}']:.1e}")
        out['models'][base] = rec
        print()

    out['peak_rss_MB'] = vm_hwm_mb()
    out['temp_after_C'] = read_temp()
    with open(os.path.join(HERE, 'bench_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"peak RSS : {out['peak_rss_MB']} MB | temp {out['temp_before_C']} -> {out['temp_after_C']} C")
    print('saved    : bench_result.json（请连同终端输出一起发回）')


if __name__ == '__main__':
    main()
