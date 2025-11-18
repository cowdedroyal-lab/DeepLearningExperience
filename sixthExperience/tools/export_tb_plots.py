#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 TensorBoard event 文件导出所有 scalar 曲线为 PNG 图片与 CSV。

使用示例：
  python tools/export_tb_plots.py --logdir outputs/segnet_camvid11/tb --outdir figures_camvid11

特性：
- 自动发现 logdir 下的 event 文件并合并同名 tag 的标量数据（按 step 排序去重）
- 为每个 scalar tag 生成独立折线图（可选滑动平均平滑）
- 将所有标量数据导出为一个 CSV，便于后续分析

依赖：tensorboard、matplotlib、pandas（CSV导出可选，若无 pandas 则回退为纯 python 写出）
"""

import argparse
import os
import sys
import glob
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
        return event_accumulator
    except Exception as e:
        print("[错误] 未安装 tensorboard，无法解析 event 文件。请先安装: pip install tensorboard", file=sys.stderr)
        raise


def load_scalars_from_events(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    """读取 logdir 下所有事件文件，并合并为 {tag: [(step, value), ...]}"""
    ea_mod = _import_event_accumulator()
    pattern = os.path.join(logdir, "events.*")
    files = sorted(glob.glob(pattern))
    if not files:
        # 兼容常见文件名
        pattern2 = os.path.join(logdir, "events.out.tfevents.*")
        files = sorted(glob.glob(pattern2))
    if not files:
        raise FileNotFoundError(f"在 {logdir} 未找到 TensorBoard event 文件")

    merged: Dict[str, Dict[int, float]] = {}
    for f in files:
        try:
            ea = ea_mod.EventAccumulator(f)
            ea.Reload()
        except Exception as e:
            print(f"[警告] 解析 {f} 失败，已跳过: {e}")
            continue
        for tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            store = merged.setdefault(tag, {})
            for ev in events:
                # 如果同一步出现，取最后一次（通常为更晚文件）
                store[int(ev.step)] = float(ev.value)
    # 转换为列表并排序
    scalars: Dict[str, List[Tuple[int, float]]] = {}
    for tag, m in merged.items():
        pairs = sorted(m.items(), key=lambda x: x[0])
        scalars[tag] = pairs
    return scalars


def moving_average(values: List[float], k: int) -> List[float]:
    if k <= 1:
        return values
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        acc += v
        if len(q) > k:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def plot_scalars(scalars: Dict[str, List[Tuple[int, float]]], outdir: str, smooth: int = 1, dpi: int = 150) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for tag, series in scalars.items():
        if not series:
            continue
        steps = [s for s, _ in series]
        vals = [v for _, v in series]
        vals_s = moving_average(vals, smooth)

        plt.figure(figsize=(6, 4), dpi=dpi)
        plt.plot(steps, vals_s, label=tag)
        plt.xlabel('Step')
        plt.ylabel(tag)
        title = tag.replace('/', ' / ')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # 清理文件名
        safe = tag.replace('/', '_').replace(' ', '_')
        png_path = os.path.join(outdir, f"{safe}.png")
        plt.savefig(png_path)
        plt.close()
        saved.append(png_path)
    return saved


def export_csv(scalars: Dict[str, List[Tuple[int, float]]], out_csv: str) -> str:
    # 统一 step 作为索引进行 outer join
    all_steps = set()
    for series in scalars.values():
        for s, _ in series:
            all_steps.add(s)
    steps_sorted = sorted(all_steps)

    # 构建二维表: rows: step, cols: tag
    tags = sorted(scalars.keys())
    table = {tag: {s: '' for s in steps_sorted} for tag in tags}
    for tag in tags:
        for s, v in scalars[tag]:
            table[tag][s] = v

    # 优先用 pandas 写出，若没有则纯 Python
    try:
        import pandas as pd
        df = pd.DataFrame({tag: [table[tag][s] for s in steps_sorted] for tag in tags}, index=steps_sorted)
        df.index.name = 'step'
        df.to_csv(out_csv)
    except Exception:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('step,' + ','.join(tags) + '\n')
            for s in steps_sorted:
                row = [str(s)] + [str(table[tag][s]) for tag in tags]
                f.write(','.join(row) + '\n')
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', required=True, help='包含 event 文件的目录')
    ap.add_argument('--outdir', required=True, help='输出图片目录')
    ap.add_argument('--smooth', type=int, default=1, help='滑动平均窗口大小，>1 启用平滑')
    ap.add_argument('--dpi', type=int, default=150, help='保存图片 DPI')
    args = ap.parse_args()

    scalars = load_scalars_from_events(args.logdir)
    if not scalars:
        print('[提示] 未在事件文件中找到任何 scalars 标签')
    saved_imgs = plot_scalars(scalars, args.outdir, smooth=args.smooth, dpi=args.dpi)
    out_csv = os.path.join(args.outdir, 'scalars.csv')
    export_csv(scalars, out_csv)

    print(f"已导出 {len(saved_imgs)} 张曲线图至: {args.outdir}")
    print(f"标量数据 CSV: {out_csv}")


if __name__ == '__main__':
    main()
