#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导出验证集的混淆矩阵：生成 CSV 与 PNG 热力图。

示例：
  python tools/export_confusion_matrix.py \
    --config configs/camvid11.yaml \
    --checkpoint outputs/segnet_camvid11/best.pt \
    --outdir figures_camvid11

说明：
- 读取配置以构建 CamVid 验证集数据加载器；
- 加载 SegNet 模型权重，在验证集上前向推理并累计混淆矩阵；
- 输出 raw 混淆矩阵（行：真实类别，列：预测类别）、以及按真实类别归一化的热力图。
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 尝试常规包导入；若失败（项目未安装为包），则回退到基于文件路径的动态导入。
try:
    from src.utils.config import load_config
    from src.models.segnet import SegNet
    from src.datasets.camvid import CamVidDataset
    from src.metrics.seg_metrics import compute_confusion_matrix, metrics_from_confusion
except ModuleNotFoundError:
    import importlib.util
    import pathlib

    ROOT = pathlib.Path(__file__).resolve().parent.parent  # 项目根目录 (sixthExperience)
    # 不修改现有目录结构，不要求增加 __init__，仅临时注入 sys.path 或按文件动态加载
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    def _load(rel_path: str):
        full = ROOT / rel_path
        name = rel_path.replace('/', '_').replace('.py', '')
        spec = importlib.util.spec_from_file_location(name, full)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    config_mod = _load('src/utils/config.py')
    models_mod = _load('src/models/segnet.py')
    dataset_mod = _load('src/datasets/camvid.py')
    metrics_mod = _load('src/metrics/seg_metrics.py')

    load_config = config_mod.load_config
    SegNet = models_mod.SegNet
    CamVidDataset = dataset_mod.CamVidDataset
    compute_confusion_matrix = metrics_mod.compute_confusion_matrix
    metrics_from_confusion = metrics_mod.metrics_from_confusion


def build_model(cfg) -> torch.nn.Module:
    model_cfg = getattr(cfg, 'model', None)
    if isinstance(model_cfg, dict):
        variant = model_cfg.get('variant', 'vanilla')
        pretrained = bool(model_cfg.get('pretrained', False))
        freeze_bn = bool(model_cfg.get('freeze_encoder_bn', False))
    else:
        variant = getattr(model_cfg, 'variant', 'vanilla') if model_cfg is not None else 'vanilla'
        pretrained = bool(getattr(model_cfg, 'pretrained', False)) if model_cfg is not None else False
        freeze_bn = bool(getattr(model_cfg, 'freeze_encoder_bn', False)) if model_cfg is not None else False
    model = SegNet(in_channels=3, num_classes=cfg.num_classes, variant=variant,
                   pretrained=pretrained, freeze_encoder_bn=freeze_bn)
    return model


def export_confusion_png(hist: np.ndarray, out_png: str, normalize: Optional[str] = None, cmap: str = 'viridis'):
    """保存混淆矩阵热力图。

    normalize: None | 'true' | 'pred'，分别表示不归一化、按真实类别行归一化或按预测列归一化。
    """
    mat = hist.astype(np.float64)
    if normalize == 'true':
        row_sums = mat.sum(axis=1, keepdims=True) + 1e-12
        mat = mat / row_sums
    elif normalize == 'pred':
        col_sums = mat.sum(axis=0, keepdims=True) + 1e-12
        mat = mat / col_sums

    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Confusion Matrix' + ('' if normalize is None else f' (normalize={normalize})'))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML 配置文件路径')
    ap.add_argument('--checkpoint', required=True, help='训练得到的 best/last.pt 路径')
    ap.add_argument('--outdir', required=True, help='输出目录，用于保存 CSV/PNG')
    ap.add_argument('--batch-size', type=int, default=None, help='验证 batch size（不指定则使用配置中的 batch_size）')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.outdir, exist_ok=True)

    # 构建验证集
    colors_path = getattr(cfg, 'class_colors_path', None)
    val_ds = CamVidDataset(
        cfg.val_images_dir, cfg.val_labels_dir, cfg.input_height, cfg.input_width,
        class_mode=cfg.class_mode, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index,
        augment=False, class_colors_path=colors_path
    )
    bs = args.batch_size or cfg.batch_size
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 模型与权重
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 累计混淆矩阵
    hist_total = None
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            hist = compute_confusion_matrix(masks, preds, cfg.num_classes, cfg.ignore_index)
            hist_total = hist if hist_total is None else hist_total + hist

    # 导出 CSV
    out_csv = os.path.join(args.outdir, 'confusion_matrix.csv')
    np.savetxt(out_csv, hist_total, fmt='%d', delimiter=',')

    # 保存 PNG（原始与按真实类别归一化）
    out_png_raw = os.path.join(args.outdir, 'confusion_matrix.png')
    out_png_norm = os.path.join(args.outdir, 'confusion_matrix_normalized_true.png')
    export_confusion_png(hist_total, out_png_raw, normalize=None)
    export_confusion_png(hist_total, out_png_norm, normalize='true')

    # 附带输出整体指标
    metrics = metrics_from_confusion(hist_total)
    print('[指标]', f"pixel_acc={metrics['pixel_acc']:.4f}", f"mean_iou={metrics['mean_iou']:.4f}")
    print('输出文件:')
    print(' -', out_csv)
    print(' -', out_png_raw)
    print(' -', out_png_norm)


if __name__ == '__main__':
    main()
