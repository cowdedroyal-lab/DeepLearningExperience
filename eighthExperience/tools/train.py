"""
训练入口脚本

功能：
- 读取 YAML 配置
- 构建 Tokenizer、Dataset/DataLoader
- 构建 CaptionTransformer 模型
- 训练循环（CrossEntropyLoss，teacher forcing）
- 每个 epoch 进行验证：val loss + BLEU-4
- 保存 best checkpoint

参考：
- Meshed-Memory Transformer 的优化器与评估节奏（本实现做了简化，教学用途）
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
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.common import load_config, set_seed, ensure_dir, save_checkpoint
from src.datasets.vocab import BertTokenizerWrapper
from src.datasets.coco_caption import build_splits, CocoCaptionDataset
from src.models.transformer_decoder import CaptionTransformer
from src.utils.metrics import compute_corpus_metrics


def evaluate(model: CaptionTransformer, loader: DataLoader, tokenizer: BertTokenizerWrapper, device: torch.device) -> Dict[str, float]:
    model.eval()
    gen: Dict[str, list] = {}
    gts: Dict[str, list] = {}
    # 使用底层 coco 来取多参考
    dataset = loader.dataset  # type: ignore
    coco = getattr(dataset, 'coco', None)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["images"].to(device)
            image_ids = batch["image_id"].tolist()
            out_ids = model.generate(images, max_len=30)
            # 对齐生成：去掉第一 token (bos)，并截断到 eos
            for i, ids in enumerate(out_ids):
                ids = ids.tolist()
                if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
                    ids = ids[1:]
                if tokenizer.eos_token_id in ids:
                    ids = ids[: ids.index(tokenizer.eos_token_id)]
                caption = tokenizer.decode(ids, skip_special_tokens=True).strip()
                img_id = str(image_ids[i])
                if img_id not in gen:  # 避免同一图片重复覆盖
                    gen[img_id] = [caption]

                # 组装 gts：从 coco 取该图所有参考
                if img_id not in gts:
                    if coco is not None:
                        ann_ids = coco.getAnnIds(imgIds=int(img_id))
                        anns = coco.loadAnns(ann_ids)
                        refs = [a["caption"] for a in anns]
                    else:
                        # 兜底：若无 coco 句柄，则至少用当前 batch 的 gt 文本
                        cur_idx = (batch["input_ids"][i]).tolist()
                        refs = [tokenizer.decode(cur_idx, skip_special_tokens=True).strip()]
                    gts[img_id] = refs

    return compute_corpus_metrics(gen, gts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/coco_baseline.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    device = torch.device(args.device)

    # Tokenizer
    tokenizer = BertTokenizerWrapper(cfg.get('tokenizer', {}))

    # Datasets & Loaders
    train_ds, val_ds, _ = build_splits(cfg, seed=cfg.get('seed', 42), tokenizer=tokenizer)
    collate = CocoCaptionDataset.collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    # Model
    model = CaptionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        fine_tune_cnn=True,
    ).to(device)

    # Optim & Loss & Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # TensorBoard 日志目录
    log_dir = os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Train Loop
    best_bleu4 = -1.0
    ensure_dir(args.save_dir)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        global_step = (epoch - 1) * len(train_loader)
        for step, batch in enumerate(pbar):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)

            logits = model(images, input_ids)  # (B, T-1, V)
            targets = input_ids[:, 1:].contiguous()  # (B, T-1)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix(loss=avg_loss)

            # TensorBoard: 记录训练 loss（按 step）
            writer.add_scalar("train/loss_iter", loss.item(), global_step + step)

        # 学习率调度并记录当前学习率（按 epoch）
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                logits = model(images, input_ids)
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        # TensorBoard: 记录 epoch 级别的 loss
        train_epoch_loss = running_loss / max(1, len(train_loader))
        writer.add_scalar("train/loss_epoch", train_epoch_loss, epoch)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)

        # Metrics (BLEU-4 等)
        metrics = evaluate(model, val_loader, tokenizer, device)
        bleu4 = metrics.get('BLEU-4', 0.0)

        # TensorBoard: 记录各类评价指标
        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, epoch)

        print({"epoch": epoch, "train_loss": train_epoch_loss, "val_loss": val_loss, **metrics})

        # Save best
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            ckpt_path = os.path.join(args.save_dir, 'coco_caption_best.pt')
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg,
                'tokenizer': cfg.get('tokenizer', {}),
                'best_bleu4': best_bleu4,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()
