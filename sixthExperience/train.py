import os
import argparse
import math
import random
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.segnet import SegNet
from src.datasets.camvid import CamVidDataset
from src.metrics.seg_metrics import evaluate_batch, compute_confusion_matrix, metrics_from_confusion
from src.utils.config import load_config
from src.utils.logger import Logger, ProgressPrinter

# 设定随机种子

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(params, cfg):
    if cfg.optimizer.lower() == 'adam':
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'sgd':
        return optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'未知优化器: {cfg.optimizer}')


def build_scheduler(optimizer, cfg):
    if cfg.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    else:
        return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='配置文件路径')
    ap.add_argument('--resume', default=None, help='恢复训练的checkpoint')
    ap.add_argument('--no-amp', action='store_true', help='禁用自动混合精度')
    return ap.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, logger, epoch) -> Dict[str, float]:
    model.train()
    printer = ProgressPrinter(total_steps=len(loader))
    total_loss = 0.0
    hist_total = None

    for step, (imgs, masks) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 若本批次全部为 ignore，则跳过，避免 CrossEntropyLoss 出现 NaN
        valid_pixels = (masks != cfg.ignore_index).sum().item()
        if valid_pixels == 0:
            # 仍然前向一次用于 BN 统计，但不反传与计入指标
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                _ = model(imgs)
            continue

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = criterion(logits, masks)
        # 训练稳定性：若出现 NaN/Inf，跳过该步
        if not torch.isfinite(loss):
            print(f"[警告] 第 {step} 步 loss 非有限值，跳过该 batch")
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪，进一步防止数值不稳定
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        # 累积混淆矩阵
        preds = torch.argmax(logits, dim=1)
        hist = compute_confusion_matrix(masks, preds, cfg.num_classes, cfg.ignore_index)
        hist_total = hist if hist_total is None else hist_total + hist

        if step % max(1, cfg.log_interval) == 0 or step == len(loader):
            metrics = metrics_from_confusion(hist)
            printer.update(step, {'loss': loss.item(), 'mIoU': metrics['mean_iou']})

    printer.done()
    epoch_loss = total_loss / len(loader)
    train_metrics = metrics_from_confusion(hist_total)
    logger.log_scalars({'loss': epoch_loss, 'pixel_acc': train_metrics['pixel_acc'], 'mean_iou': train_metrics['mean_iou']}, epoch, prefix='train')
    return {'loss': epoch_loss, **train_metrics}


def evaluate(model, loader, device, cfg, logger, epoch, split='val') -> Dict[str, float]:
    model.eval()
    hist_total = None
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            hist = compute_confusion_matrix(masks, preds, cfg.num_classes, cfg.ignore_index)
            hist_total = hist if hist_total is None else hist_total + hist
    metrics = metrics_from_confusion(hist_total)
    logger.log_scalars({'pixel_acc': metrics['pixel_acc'], 'mean_iou': metrics['mean_iou']}, epoch, prefix=split)
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    save_root = os.path.join(cfg.save_dir, cfg.experiment_name)
    os.makedirs(save_root, exist_ok=True)

    logger = Logger(os.path.join(save_root, 'tb'))
    logger.log_text('config', str(cfg.data))

    # 数据集
    # 若提供 class_colors_path（用于 RGB 标签颜色 -> 索引映射，主要在 camvid_32 模式下），则传入数据集
    colors_path = getattr(cfg, 'class_colors_path', None)
    train_ds = CamVidDataset(
        cfg.train_images_dir, cfg.train_labels_dir, cfg.input_height, cfg.input_width,
        class_mode=cfg.class_mode, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index,
        augment=True, class_colors_path=colors_path
    )
    val_ds = CamVidDataset(
        cfg.val_images_dir, cfg.val_labels_dir, cfg.input_height, cfg.input_width,
        class_mode=cfg.class_mode, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index,
        augment=False, class_colors_path=colors_path
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 模型
    # 允许通过配置文件选择 SegNet 变体及预训练选项
    model_cfg = getattr(cfg, 'model', None)
    if model_cfg is None:
        # 兼容旧配置：默认为 vanilla 结构、无预训练
        variant = 'vanilla'
        pretrained = False
        freeze_bn = False
    else:
        # 兼容两种写法：cfg.model 既可能是 dict，也可能是带属性的对象
        if isinstance(model_cfg, dict):
            variant = model_cfg.get('variant', 'vanilla')
            pretrained = bool(model_cfg.get('pretrained', False))
            freeze_bn = bool(model_cfg.get('freeze_encoder_bn', False))
        else:
            # 使用 getattr 防止缺字段时报错，保持工程稳健
            variant = getattr(model_cfg, 'variant', 'vanilla')
            pretrained = bool(getattr(model_cfg, 'pretrained', False))
            freeze_bn = bool(getattr(model_cfg, 'freeze_encoder_bn', False))

    print(f"构建 SegNet 模型: variant={variant}, pretrained={pretrained}, freeze_encoder_bn={freeze_bn}")

    model = SegNet(
        in_channels=3,
        num_classes=cfg.num_classes,
        variant=variant,
        pretrained=pretrained,
        freeze_encoder_bn=freeze_bn,
    )
    model.to(device)

    # 损失 (可加入类权重)
    class_weights: Optional[torch.Tensor] = None
    if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
        if isinstance(cfg.class_weights, (list, tuple)):
            if len(cfg.class_weights) != cfg.num_classes:
                raise ValueError('class_weights 长度必须等于 num_classes')
            # 将权重放到与模型相同的设备，避免 CE 报 device mismatch
            class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
        else:
            print('[警告] class_weights 配置非列表，已忽略')
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index)

    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and cfg.amp and device.type == 'cuda'))

    best_miou = -1.0
    start_epoch = 0

    # 恢复训练
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_miou = ckpt.get('best_miou', best_miou)
        print(f'从 {args.resume} 恢复，起始 epoch = {start_epoch}, best_mIoU = {best_miou:.4f}')

    for epoch in range(start_epoch, cfg.epochs):
        print(f'=== Epoch {epoch}/{cfg.epochs - 1} ===')
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, logger, epoch)
        if scheduler:
            scheduler.step()

        if (epoch % cfg.val_interval == 0) or (epoch == cfg.epochs - 1):
            val_stats = evaluate(model, val_loader, device, cfg, logger, epoch, split='val')
            miou = val_stats['mean_iou']
            is_best = miou > best_miou
            if is_best:
                best_miou = miou
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': best_miou,
            }
            if scheduler:
                ckpt['scheduler'] = scheduler.state_dict()
            if scaler:
                ckpt['scaler'] = scaler.state_dict()
            torch.save(ckpt, os.path.join(save_root, 'last.pt'))
            if is_best:
                torch.save(ckpt, os.path.join(save_root, 'best.pt'))
                print(f'[保存最佳模型] mIoU={best_miou:.4f}')

    logger.close()

if __name__ == '__main__':
    main()
