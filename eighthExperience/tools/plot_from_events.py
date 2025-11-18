#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional

import matplotlib

# 使用无头环境绘图
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)


def ema_smooth(values: List[float], alpha: float) -> List[float]:
    if not values or alpha <= 0:
        return values
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return smoothed


def load_event_accumulator(path: str, size_guidance: Optional[Dict] = None) -> EventAccumulator:
    if size_guidance is None:
        size_guidance = {
            "scalars": 0,  # 0 表示加载全部
            "images": 0,
            "histograms": 0,
        }
    ea = EventAccumulator(path, size_guidance=size_guidance)
    ea.Reload()
    return ea


def filter_tags(tags: Iterable[str], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    tags = list(tags)
    if include:
        inc_re = [re.compile(p) for p in include]
        tags = [t for t in tags if any(r.search(t) for r in inc_re)]
    if exclude:
        exc_re = [re.compile(p) for p in exclude]
        tags = [t for t in tags if not any(r.search(t) for r in exc_re)]
    # 常见噪声标签可默认排除
    noise_patterns = [
        r"learning_rate.*/bucket_\d+",
        r"wall_time|global_step/sec|examples_per_second",
    ]
    tags = [t for t in tags if not any(re.search(p, t) for p in noise_patterns)]
    tags.sort()
    return tags


def plot_scalar(tag: str, steps: List[int], values: List[float], out_path: str, *, title: Optional[str] = None, dpi: int = 160):
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, label=tag, linewidth=1.8)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.title(title or tag)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="从 TensorBoard event 文件提取标量并绘图 (per-tag PNG)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--logdir", type=str, default=os.path.join("runs", "log_epoch20_old"),
                       help="包含事件文件的目录（单个 run）")
    group.add_argument("--event", type=str, default=None, help="单个事件文件路径")
    parser.add_argument("--include", nargs="*", default=None,
                        help="只包含匹配这些正则的标签，如: loss acc bleu")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="排除匹配这些正则的标签")
    parser.add_argument("--outdir", type=str, default=None,
                        help="输出目录，默认在输入目录下创建 plots/")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="EMA 平滑系数 alpha ∈ [0,1]，0 表示不平滑")
    parser.add_argument("--dpi", type=int, default=160, help="输出图片 DPI")

    args = parser.parse_args()

    # 解析输入路径
    input_path = args.event or args.logdir
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    # 输出目录
    if args.outdir:
        outdir = args.outdir
    else:
        base = args.logdir if args.logdir else os.path.dirname(args.event)
        outdir = os.path.join(base, "plots")
    os.makedirs(outdir, exist_ok=True)

    print(f"加载事件: {input_path}")
    ea = load_event_accumulator(input_path)
    tags_all = ea.Tags().get("scalars", [])
    if not tags_all:
        raise RuntimeError("未在事件文件中发现任何 scalar 标签。")

    tags = filter_tags(tags_all, args.include, args.exclude)
    if not tags:
        raise RuntimeError("筛选后没有可绘制的标签，请调整 --include/--exclude。")

    print("发现的 scalar 标签（将绘制）:")
    for t in tags:
        print(" -", t)

    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        vals = [float(e.value) for e in events]
        if args.smoothing and args.smoothing > 0:
            vals_plot = ema_smooth(vals, args.smoothing)
            title = f"{tag} (EMA α={args.smoothing})"
        else:
            vals_plot = vals
            title = tag

        safe_tag = re.sub(r"[^\w\-_.]+", "_", tag)
        out_path = os.path.join(outdir, f"{safe_tag}.png")
        plot_scalar(tag, steps, vals_plot, out_path, title=title, dpi=args.dpi)
        print(f"保存: {out_path}")

    print(f"完成，输出目录: {outdir}")


if __name__ == "__main__":
    main()
