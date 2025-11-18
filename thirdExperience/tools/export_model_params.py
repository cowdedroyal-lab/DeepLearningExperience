#!/usr/bin/env python3
"""
从保存的 checkpoint 中解析模型的基本信息并输出为 Markdown 表格。

示例：
  python tools/export_model_params.py \
    --checkpoint checkpoints/best_model.pth \
    --output thirdExperience/figures/model_params.md
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch


def parse_model_info_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # 词表与嵌入维度
    emb_w = state_dict.get('embeddings.weight')
    if emb_w is not None:
        vocab_size, embedding_dim = emb_w.shape
        info['vocab_size'] = int(vocab_size)
        info['embedding_dim'] = int(embedding_dim)
    else:
        info['vocab_size'] = '未知'
        info['embedding_dim'] = '未知'

    # LSTM 层数、隐藏维度、是否双向
    # 通过权重命名约定推断，例如：lstm.weight_hh_l0, lstm.weight_ih_l1, lstm.weight_ih_l0_reverse
    hidden_dim = None
    max_layer_idx = -1
    bidirectional = False
    for k, v in state_dict.items():
        if k.startswith('lstm.weight_hh_l') and 'reverse' not in k:
            # 形状为 [4*hidden_dim, hidden_dim]
            hidden_dim = int(v.shape[1])
            try:
                layer_idx = int(k.split('l')[-1])
                max_layer_idx = max(max_layer_idx, layer_idx)
            except Exception:
                pass
        if 'reverse' in k:
            bidirectional = True

    if hidden_dim is not None:
        info['hidden_dim'] = hidden_dim
    else:
        info['hidden_dim'] = '未知'

    info['num_layers'] = int(max_layer_idx + 1) if max_layer_idx >= 0 else '未知'
    info['bidirectional'] = bool(bidirectional)

    # 是否 weight tying
    has_linear = any(k.startswith('linear.') for k in state_dict.keys())
    has_proj = any(k.startswith('proj.') for k in state_dict.keys())
    has_out_bias = any(k == 'output_bias' for k in state_dict.keys())
    if has_proj or has_out_bias:
        info['weight_tying'] = True
    elif has_linear:
        info['weight_tying'] = False
    else:
        info['weight_tying'] = '未知'

    # 参数总量
    total_params = sum(int(t.numel()) for t in state_dict.values())
    info['total_params'] = total_params

    return info


def write_markdown_table(info: Dict[str, Any], out_path: str) -> None:
    lines = []
    lines.append('# 模型基本信息\n')
    lines.append('\n')
    lines.append('| 项目 | 数值 |\n')
    lines.append('|---|---:|\n')
    lines.append(f"| 词表大小 (vocab_size) | {info.get('vocab_size','-')} |\n")
    lines.append(f"| 嵌入维度 (embedding_dim) | {info.get('embedding_dim','-')} |\n")
    lines.append(f"| 隐藏维度 (hidden_dim) | {info.get('hidden_dim','-')} |\n")
    lines.append(f"| LSTM 层数 (num_layers) | {info.get('num_layers','-')} |\n")
    lines.append(f"| 双向 LSTM (bidirectional) | {info.get('bidirectional','-')} |\n")
    lines.append(f"| 权重共享 (weight_tying) | {info.get('weight_tying','-')} |\n")
    lines.append(f"| 参数总量 (total_params) | {info.get('total_params','-'):,} |\n")
    lines.append('\n')
    # 训练超参数（来自代码默认值，若实际训练修改请以实际为准）
    lines.append('> 说明：Dropout、优化器类型、学习率等训练超参数以代码默认值或运行参数为准；如实验中有调整，请以命令行参数为准。\n')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description='从 checkpoint 提取模型信息并生成 Markdown 表格')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='模型权重文件路径')
    parser.add_argument('--output', type=str, default='figures/model_params.md', help='Markdown 输出路径')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'未找到 checkpoint: {ckpt_path}')

    state = torch.load(str(ckpt_path), map_location='cpu')
    # 支持直接 state_dict 或 包含 state_dict 的字典
    if isinstance(state, dict) and not any(isinstance(v, torch.Tensor) for v in state.values()):
        # 例如 {'model_state_dict': ..., 'optimizer_state_dict': ...}
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']
        else:
            # fallback：尝试找到第一个 tensor dict
            for v in state.values():
                if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                    state = v
                    break

    if not isinstance(state, dict):
        raise RuntimeError('无法解析 checkpoint 中的 state_dict')

    info = parse_model_info_from_state_dict(state)
    write_markdown_table(info, args.output)
    print(f"已生成模型信息表: {args.output}")


if __name__ == '__main__':
    main()
