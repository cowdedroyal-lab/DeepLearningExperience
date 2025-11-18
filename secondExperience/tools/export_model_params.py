import os
import argparse
import torch
from typing import List, Tuple


def load_state_dict(checkpoint_dir: str, net: str):
    ckpt_path = os.path.join(checkpoint_dir, f'{net}-best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'net' in ckpt:
        return ckpt['net']
    # fallback: whole dict is state_dict
    return ckpt


def state_dict_to_rows(state_dict: dict) -> Tuple[List[Tuple[str, str, int]], int]:
    rows: List[Tuple[str, str, int]] = []
    total_params = 0
    for name, tensor in state_dict.items():
        numel = tensor.numel()
        shape = list(tensor.shape)
        total_params += numel
        rows.append((name, 'x'.join(map(str, shape)) if shape else 'scalar', numel))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows, total_params


def main():
    parser = argparse.ArgumentParser(description='Export model parameter table from checkpoint state_dict.')
    parser.add_argument('--checkpoint-dir', default='checkpoint')
    parser.add_argument('--net', default='vit_cifar10_optimized')
    parser.add_argument('--output', default='figures/model_params.md')
    parser.add_argument('--topk', type=int, default=50, help='Limit rows to top-K by parameter count (0=no limit)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sd = load_state_dict(args.checkpoint_dir, args.net)
    rows, total = state_dict_to_rows(sd)
    if args.topk and args.topk > 0:
        show_rows = rows[:args.topk]
    else:
        show_rows = rows

    # Write Markdown table manually (avoid pandas dependency)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"### 模型参数统计\n\n")
        f.write(f"总参数量: {total:,}\n\n")
        f.write("| 参数名 | 形状 | 数量 |\n")
        f.write("|---|---:|---:|\n")
        for name, shape, numel in show_rows:
            f.write(f"| {name} | {shape} | {numel:,} |\n")
        if len(rows) > len(show_rows):
            f.write(f"\n\n> 注：仅展示前 {len(show_rows)} 行，更多参数见 checkpoint state_dict。\n")

    print('Saved markdown table to', args.output)

if __name__ == '__main__':
    main()
