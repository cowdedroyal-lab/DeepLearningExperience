"""
模型评估脚本

功能：
- 加载训练好的模型
- 指定 split=val/test
- 使用 greedy 生成字幕
- 计算 BLEU/METEOR/ROUGE-L/CIDEr（简化实现）
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import load_config, set_seed
from src.datasets.vocab import BertTokenizerWrapper
from src.datasets.word_vocab import WordVocab
from src.datasets.coco_caption import build_splits, CocoCaptionDataset
from src.models.transformer_decoder import CaptionTransformer
from src.utils.metrics import compute_corpus_metrics
from src.utils.coco_official_eval import coco_official_eval


def build_split(cfg: Dict[str, Any], split: str, tokenizer: BertTokenizerWrapper):
    train_ds, val_ds, test_ds = build_splits(cfg, seed=cfg.get('seed', 42), tokenizer=tokenizer)
    if split == 'val':
        return val_ds
    elif split == 'test':
        return test_ds
    else:
        raise ValueError("split must be 'val' or 'test'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/coco_baseline.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = torch.device(args.device)

    # 从 checkpoint 恢复 tokenizer 优先，保证与训练一致
    state = torch.load(args.checkpoint, map_location=device)
    saved_tok_cfg = state.get('tokenizer', {}) if isinstance(state, dict) else {}
    tok_type = (saved_tok_cfg.get('type') or cfg.get('tokenizer', {}).get('type') or 'bert').lower()
    if tok_type == 'word' or 'word_map' in saved_tok_cfg:
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

    ds = build_split(cfg, args.split, tokenizer)
    collate = CocoCaptionDataset.collate_fn(tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

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

    # 生成
    gen: Dict[str, list] = {}
    gts: Dict[str, list] = {}
    coco = getattr(ds, 'coco', None)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval-{args.split}"):
            images = batch['images'].to(device)
            image_ids = batch['image_id'].tolist()
            out_ids = model.generate(images, max_len=30)
            for i, ids in enumerate(out_ids):
                ids = ids.tolist()
                if len(ids) > 0 and ids[0] == bos_id:
                    ids = ids[1:]
                if eos_id in ids:
                    ids = ids[: ids.index(eos_id)]
                caption = tokenizer.decode(ids, skip_special_tokens=True).strip()
                img_id = str(image_ids[i])
                if img_id not in gen:
                    gen[img_id] = [caption]
                if img_id not in gts:
                    if coco is not None:
                        ann_ids = coco.getAnnIds(imgIds=int(img_id))
                        anns = coco.loadAnns(ann_ids)
                        refs = [a['caption'] for a in anns]
                    else:
                        cur_idx = (batch['input_ids'][i]).tolist()
                        refs = [tokenizer.decode(cur_idx, skip_special_tokens=True).strip()]
                    gts[img_id] = refs

    # 优先官方评估（若可用）
    if coco is not None:
        try:
            preds_map = {k: v[0] for k, v in gen.items()}
            scores = coco_official_eval(coco, preds_map)
        except Exception as e:
            print(f"[Warn] 官方 COCO 评估失败，回退到简化实现: {e}")
            scores = compute_corpus_metrics(gen, gts)
    else:
        scores = compute_corpus_metrics(gen, gts)
    print(scores)


if __name__ == '__main__':
    main()
