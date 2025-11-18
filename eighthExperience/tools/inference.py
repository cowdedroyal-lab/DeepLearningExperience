"""
单张图片字幕生成脚本

功能：
- 加载训练好的 CaptionTransformer 模型
- 对单张图片生成字幕（greedy）
- 输出到控制台，或可选写入文件
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any

# 确保可以以模块形式导入 src/*
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from PIL import Image

from src.utils.common import load_config, set_seed
from src.datasets.vocab import BertTokenizerWrapper
from src.datasets.word_vocab import WordVocab  # 词表分词支持（与训练保持一致）
from src.datasets.coco_caption import build_default_transforms
from src.models.transformer_decoder import CaptionTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/coco_baseline.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--output', type=str, default=None, help='可选：将字幕写入到该路径')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = torch.device(args.device)

    # 先加载 checkpoint 以获取保存的 tokenizer 配置/词表
    state = torch.load(args.checkpoint, map_location=device)
    saved_tok_cfg = state.get('tokenizer', {}) if isinstance(state, dict) else {}

    # Tokenizer 选择：优先使用 checkpoint 中的类型与词表
    tok_type = (saved_tok_cfg.get('type') or cfg.get('tokenizer', {}).get('type') or 'bert').lower()
    if tok_type == 'word' or 'word_map' in saved_tok_cfg:
        # 从 checkpoint 恢复词表，确保与训练时一致
        word_map = saved_tok_cfg.get('word_map')
        if not word_map:
            raise RuntimeError('当前 checkpoint 未保存 word_map，无法恢复词表分词。请使用保存了 tokenizer.word_map 的模型或切换到 BERT tokenizer。')
        inv = {int(i): w for w, i in word_map.items()} if all(isinstance(v, int) for v in word_map.values()) else {int(v): k for k, v in word_map.items()}
        tokenizer = WordVocab(word_map=word_map, inv_map=inv)
        bos_id = tokenizer.start_token_id
        eos_id = tokenizer.end_token_id
    else:
        tokenizer = BertTokenizerWrapper(cfg.get('tokenizer', {}))
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

    # Transform
    resize = tuple(cfg.get('transforms', {}).get('resize', [224, 224]))
    imagenet_norm = bool(cfg.get('transforms', {}).get('imagenet_normalize', True))
    transform = build_default_transforms(resize=resize, imagenet_normalize=imagenet_norm)

    # Model
    model = CaptionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=tokenizer.pad_token_id,
        bos_id=bos_id,
        eos_id=eos_id,
        fine_tune_cnn=True,
    ).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    # Load image
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    img = Image.open(args.image).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_ids = model.generate(img_t, max_len=args.max_len)
        ids = out_ids[0].tolist()
        if len(ids) > 0 and ids[0] == bos_id:
            ids = ids[1:]
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        caption = tokenizer.decode(ids, skip_special_tokens=True).strip()

    print(caption)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(caption + "\n")


if __name__ == '__main__':
    main()
