# -*- coding: utf-8 -*-
"""打包补充材料 zip：以 revision_experiments/ 为根，逐文件与仓库对齐。

用法（仓库根目录）：
    python revision_experiments/build_supp_zip.py [输出路径]

默认输出到仓库上一级的 `MIMO-WM-补充材料-CCTA260363.zip`。

排除项仅限运行痕迹，不含实验产物，且与仓库 .gitignore 保持一致：
    logs/                 运行日志
    __pycache__/          字节码缓存
    results/_backup_cpu/  某次调试前的中间备份
    *.log                 运行日志（仓库 .gitignore 第 30 行已忽略）

另有一类**只影响投稿 zip、不影响仓库**的收窄：修改稿表 6 与第 5.7 节只报告 MIMO-WM
自身的机载数值，因此 zip 只随附 MIMO-WM 的 ONNX 与逐窗口结果；其余模型的 ONNX 与同批
机载结果保留在仓库中，可用同一套脚本重新导出与测量，但不随投稿材料分发。见
ZIP_ONLY_MIMO。

打完包可用 --check 只做比对不写文件，用于确认 zip 与工作区没有版本漂移。
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(os.path.dirname(REPO), 'MIMO-WM-补充材料-CCTA260363.zip')
SKIP_DIRS = {'__pycache__', 'logs', '_backup_cpu'}

# 机载目录中只随 zip 分发 MIMO-WM 自身的产物（键为相对本目录的目录名）
ZIP_ONLY_MIMO = {
    'onboard_bench/models': lambda n: n == 'manifest.json' or n.startswith('MIMO-WM'),
    'onboard_bench/results': lambda n: (n.startswith('bench_result_MIMO-WM')
                                        or n == 'bench_result_mpc.json'),
}


def keep(rel):
    rel = rel.replace(os.sep, '/')
    parts = rel.split('/')
    if any(p in SKIP_DIRS for p in parts):
        return False
    if parts[-1].endswith('.log'):
        return False
    only = ZIP_ONLY_MIMO.get('/'.join(parts[:-1]))
    return only is None or only(parts[-1])


def collect():
    """返回 [(绝对路径, zip 内相对路径)]，按 zip 内路径排序。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(HERE):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, REPO)
            if keep(os.path.relpath(full, HERE)):
                out.append((full, rel.replace(os.sep, '/')))
    out.sort(key=lambda x: x[1])
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0] if args else DEFAULT_OUT
    files = collect()

    if '--check' in sys.argv:
        have = set(zipfile.ZipFile(out).namelist()) if os.path.exists(out) else set()
        want = {rel for _, rel in files}
        print('源目录有 zip缺:', sorted(want - have))
        print('zip有 源目录无:', sorted(have - want))
        print('一致' if want == have else '不一致')
        return 0 if want == have else 1

    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, rel in files:
            z.write(full, rel)

    raw = sum(os.path.getsize(f) for f, _ in files)
    print('%d files, %.2f MB raw -> %.2f MB zip' % (len(files), raw / 1e6,
                                                    os.path.getsize(out) / 1e6))
    print('-> %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
