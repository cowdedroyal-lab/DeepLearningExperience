#!/usr/bin/env python3
"""
导出 TensorBoard 事件文件中的标量曲线为图片。

用法示例：
  python tools/export_tb_plots.py \
    --event-file runs/poetry_gen_1762306872/events.out.tfevents.1762306872.vpdlab-X640-G40.656888.0 \
    --output-dir figures

如果提供的是 --logdir，将自动选择 logdir 下最新的事件文件。
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_latest_event_file(logdir: str) -> str:
    p = Path(logdir)
    if not p.exists():
        raise FileNotFoundError(f"logdir 不存在: {logdir}")
    event_files: List[Path] = list(p.rglob('events.out.tfevents.*'))
    if not event_files:
        raise FileNotFoundError(f"在 {logdir} 未找到任何 TensorBoard 事件文件")
    # 以修改时间排序，取最新
    event_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(event_files[0])


def safe_name(tag: str) -> str:
    return tag.replace('/', '_').replace(' ', '_')


def plot_scalar(tag: str, steps: List[int], values: List[float], out_path: str,
                xlabel: str = 'Step', ylabel: str = 'Value') -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, label=tag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(tag)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='导出 TensorBoard 标量曲线为 PNG')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--event-file', type=str, help='具体事件文件路径')
    g.add_argument('--logdir', type=str, help='包含事件文件的目录，将自动选择最新的事件文件')
    parser.add_argument('--output-dir', type=str, default='figures', help='输出图片目录')
    args = parser.parse_args()

    event_file = args.event_file or find_latest_event_file(args.logdir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载事件文件: {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        print('未找到任何标量数据。')
        return

    print('发现标量标签:')
    for t in scalar_tags:
        print(f' - {t}')

    # 单独导出每个 tag
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        xlabel = 'Epoch' if 'validation' in tag or 'Perplexity' in tag else 'Step'
        ylabel = 'Perplexity' if 'Perplexity' in tag else 'Value'
        out_path = os.path.join(args.output_dir, f'{safe_name(tag)}.png')
        plot_scalar(tag, steps, values, out_path, xlabel=xlabel, ylabel=ylabel)

    # 组合常见曲线
    common_pairs = [
        (['Loss/train', 'Loss/validation'], 'Loss curves', 'Step/Epoch', 'Loss', 'Loss_combined.png'),
        (['Perplexity/validation'], 'Validation Perplexity', 'Epoch', 'Perplexity', 'Perplexity_validation.png'),
        (['GradNorm/train'], 'Training Gradient Norm', 'Step', 'Grad Norm', 'GradNorm_train.png'),
    ]

    for tags, title, xlabel, ylabel, filename in common_pairs:
        available = [t for t in tags if t in scalar_tags]
        if not available:
            continue
        plt.figure(figsize=(7, 4))
        for tag in available:
            ev = ea.Scalars(tag)
            steps = [e.step for e in ev]
            values = [e.value for e in ev]
            plt.plot(steps, values, label=tag)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, filename), dpi=150)
        plt.close()

    print(f"导出完成，图片已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
