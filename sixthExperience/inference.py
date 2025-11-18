import os
import argparse
import torch
from PIL import Image
from typing import List

from src.models.segnet import SegNet
from src.utils.visualize import save_color_mask
from src.utils.config import load_config
from src.datasets.camvid import CamVidDataset, build_color_to_index

import torchvision.transforms as T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--input', required=True, help='单张图像或目录')
    ap.add_argument('--output', required=True)
    ap.add_argument('--palette', default=None, help='暂不实现自定义文件，可用默认')
    return ap.parse_args()


def load_image(path, size):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img)


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

    print(f"[Inference] 构建 SegNet 模型: variant={variant}, pretrained={pretrained}, freeze_encoder_bn={freeze_bn}")

    # 推理阶段不再额外加载 ImageNet 预训练，完全依赖 checkpoint
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
        print('[Inference] 严格加载 state_dict 失败，尝试 strict=False，错误信息:', e)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    size = (cfg.input_height, cfg.input_width)

    # 构建可视化调色板：
    # - 若是 32 类且提供了 class_colors_path，则按 JSON 中 id->color 反向构建一致的调色板
    # - 否则沿用默认（11类使用默认11色，其它随机色）
    palette = None
    colors_path = getattr(cfg, 'class_colors_path', None)
    if getattr(cfg, 'class_mode', None) == 'camvid_32' and colors_path and os.path.isfile(colors_path):
        import json
        with open(colors_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        id2color = {}
        if isinstance(data, list):
            for item in data:
                if 'id' in item and 'color' in item:
                    cid = int(item['id'])
                    col = tuple(int(x) for x in item['color'])
                    # 跳过 ignore_index (如 255)
                    if 0 <= cid < cfg.num_classes:
                        id2color[cid] = col
        elif isinstance(data, dict):
            for k, v in data.items():
                parts = [int(x) for x in k.split(',')]
                cid = int(v)
                if 0 <= cid < cfg.num_classes and len(parts) == 3:
                    id2color[cid] = (parts[0], parts[1], parts[2])
        # 生成按类别索引顺序的调色板
        if id2color:
            palette = [id2color.get(i, (0, 0, 0)) for i in range(cfg.num_classes)]
    inp_path = args.input
    if os.path.isdir(inp_path):
        img_files = [os.path.join(inp_path, f) for f in os.listdir(inp_path) if f.lower().endswith(('jpg','png'))]
    else:
        img_files = [inp_path]

    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for f in img_files:
            img_tensor = load_image(f, size).unsqueeze(0).to(device)
            logits = model(img_tensor)
            pred = torch.argmax(logits, dim=1)[0]  # (H W)
            out_name = os.path.splitext(os.path.basename(f))[0] + '_pred.png'
            save_color_mask(pred.cpu(), os.path.join(args.output, out_name), cfg.num_classes, palette=palette)
            print('保存预测:', out_name)

if __name__ == '__main__':
    main()
