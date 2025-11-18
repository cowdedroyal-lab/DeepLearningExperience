import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.models.segnet import SegNet
from src.datasets.camvid import CamVidDataset
from src.metrics.seg_metrics import compute_confusion_matrix, metrics_from_confusion
from src.utils.config import load_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='val', choices=['val','test'])
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 按配置构建与训练阶段一致的 SegNet 变体
    model_cfg = getattr(cfg, 'model', None)
    if model_cfg is None:
        variant = 'vanilla'
        pretrained = False
        freeze_bn = False
    else:
        if isinstance(model_cfg, dict):
            variant = model_cfg.get('variant', 'vanilla')
            pretrained = bool(model_cfg.get('pretrained', False))
            freeze_bn = bool(model_cfg.get('freeze_encoder_bn', False))
        else:
            variant = getattr(model_cfg, 'variant', 'vanilla')
            pretrained = bool(getattr(model_cfg, 'pretrained', False))
            freeze_bn = bool(getattr(model_cfg, 'freeze_encoder_bn', False))

    print(f"[Eval] 构建 SegNet 模型: variant={variant}, pretrained={pretrained}, freeze_encoder_bn={freeze_bn}")

    # 评估阶段不再单独加载 ImageNet 预训练，完全依赖 checkpoint
    model = SegNet(
        in_channels=3,
        num_classes=cfg.num_classes,
        variant=variant,
        pretrained=False,
        freeze_encoder_bn=freeze_bn,
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print('[Eval] 严格加载 state_dict 失败，尝试 strict=False，错误信息:', e)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    if args.split == 'val':
        images_dir = cfg.val_images_dir
        labels_dir = cfg.val_labels_dir
    else:
        images_dir = cfg.test_images_dir
        labels_dir = cfg.test_labels_dir

    colors_path = getattr(cfg, 'class_colors_path', None)
    ds = CamVidDataset(
        images_dir, labels_dir, cfg.input_height, cfg.input_width,
        class_mode=cfg.class_mode, num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index, augment=False, class_colors_path=colors_path
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

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
    print('Pixel Acc:', metrics['pixel_acc'])
    print('Mean Pixel Acc:', metrics['mean_pixel_acc'])
    print('Mean IoU:', metrics['mean_iou'])
    print('Class IoU:', metrics['class_iou'])

if __name__ == '__main__':
    main()
