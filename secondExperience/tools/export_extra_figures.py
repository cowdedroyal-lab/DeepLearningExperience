import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import matplotlib.pyplot as plt


def choose_latest_run(log_dir: str, net: str):
    if not os.path.isdir(log_dir):
        return None
    candidates = []
    for d in os.listdir(log_dir):
        if d.startswith(net):
            p = os.path.join(log_dir, d)
            if os.path.isdir(p):
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_scalars(run_dir: str):
    acc = EventAccumulator(run_dir, size_guidance={'scalars':0})
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        if tag.startswith('Loss/') or tag.startswith('Accuracy/'):
            events = acc.Scalars(tag)
            data[tag] = sorted([(e.step, e.value) for e in events], key=lambda x: x[0])
    return data


def plot_generalization_gap(scalars: dict, out_path: str):
    acc_train = scalars.get('Accuracy/train') or scalars.get('Accuracy/Train')
    acc_test = scalars.get('Accuracy/test') or scalars.get('Accuracy/Test')
    if not acc_train or not acc_test:
        print('Missing Accuracy/train or Accuracy/test for gap plot')
        return None
    # align by step
    test_dict = {s:v for s,v in acc_test}
    gap = []
    for s,v in acc_train:
        if s in test_dict:
            gap.append((s, test_dict[s]-v))
    if not gap:
        print('No overlapping steps for gap plot')
        return None
    steps = [s for s,_ in gap]
    values = [g for _,g in gap]
    plt.figure(figsize=(6,4))
    plt.plot(steps, values, color='purple')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel('Step')
    plt.ylabel('Test - Train Accuracy (pp)')
    plt.title('Generalization Gap Over Training')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_weight_hist(checkpoint_path: str, out_path: str):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint for weight histogram')
        return None
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    import numpy as np
    all_w = []
    for k,t in state.items():
        if t.dtype.is_floating_point:
            all_w.append(t.detach().cpu().view(-1).numpy())
    if not all_w:
        print('No floating weights collected')
        return None
    arr = np.concatenate(all_w)
    plt.figure(figsize=(6,4))
    plt.hist(arr, bins=120, color='#1f77b4', alpha=0.85)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution (All Layers)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Export extra analytical figures for report')
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--checkpoint', default='checkpoint/vit_cifar10_optimized-best.pth')
    parser.add_argument('--net', default='vit_cifar10_optimized')
    parser.add_argument('--figures-dir', default='figures')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    run_dir = choose_latest_run(args.log_dir, args.net)
    outputs = {}
    if run_dir:
        scalars = load_scalars(run_dir)
        gap_path = os.path.join(args.figures_dir, 'generalization_gap.png')
        p_gap = plot_generalization_gap(scalars, gap_path)
        if p_gap: outputs['generalization_gap'] = p_gap
    else:
        print('No run directory found for extra figures')

    w_hist_path = os.path.join(args.figures_dir, 'weight_histogram.png')
    p_w = plot_weight_hist(args.checkpoint, w_hist_path)
    if p_w: outputs['weight_histogram'] = p_w

    print('Generated:', outputs)

if __name__ == '__main__':
    main()
