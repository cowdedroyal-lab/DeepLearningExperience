import os
import argparse
import json
from typing import List
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def find_event_files(log_dir: str) -> List[str]:
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))
    return sorted(event_files)


def extract_scalars_from_run(run_dir: str):
    # 使用 TensorBoard 的 EventAccumulator 解析标量，无需 TensorFlow
    acc = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        if tag.startswith('Loss/') or tag.startswith('Accuracy/'):
            events = acc.Scalars(tag)
            data[tag] = sorted([(e.step, e.value) for e in events], key=lambda x: x[0])
    return data


def plot_metrics(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Group by prefix (Loss, Accuracy)
    grouped = {}
    for tag, series in data.items():
        prefix = tag.split('/')[0]
        grouped.setdefault(prefix, []).append((tag, series))

    saved = []
    for prefix, items in grouped.items():
        plt.figure(figsize=(6,4))
        for tag, series in items:
            steps = [s for s,_ in series]
            values = [v for _,v in series]
            plt.plot(steps, values, label=tag)
        plt.xlabel('Step')
        plt.ylabel(prefix)
        plt.title(f'{prefix} curves')
        plt.legend()
        out_path = os.path.join(output_dir, f'{prefix.lower()}_curves.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard scalar plots to PNG.')
    parser.add_argument('--log-dir', default='logs', help='Root log directory containing event files.')
    parser.add_argument('--net', default='vit_cifar10_optimized', help='Network name used in log folder prefix.')
    parser.add_argument('--output-dir', default='figures', help='Directory to save plots.')
    args = parser.parse_args()

    # locate latest run matching net
    target_prefix = args.net
    candidate_runs = []
    if os.path.isdir(args.log_dir):
        for d in os.listdir(args.log_dir):
            if d.startswith(target_prefix):
                full = os.path.join(args.log_dir, d)
                if os.path.isdir(full):
                    candidate_runs.append(full)
    if not candidate_runs:
        print('No matching log runs found.')
        return
    # choose most recent by mtime
    candidate_runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    run_dir = candidate_runs[0]
    print('Using log run:', run_dir)

    event_files = find_event_files(run_dir)
    if not event_files:
        print('No event files found.')
        return

    data = extract_scalars_from_run(run_dir)
    saved = plot_metrics(data, args.output_dir)
    print('Saved figures:', json.dumps(saved, indent=2))

if __name__ == '__main__':
    main()
