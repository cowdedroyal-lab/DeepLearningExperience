import os
from typing import Tuple, Dict
import numpy as np
from PIL import Image
import torch

# 可视化预测结果，将类别索引映射到颜色
# 用户可传入自定义颜色表，否则使用简单随机/或默认 CamVid 11

DEFAULT_PALETTE_11 = [
    (128,128,128),(128,0,0),(192,192,128),(128,64,128),(0,0,192),
    (128,128,0),(192,128,128),(64,64,128),(64,0,128),(64,128,128),(192,64,128)
]

def ensure_palette(num_classes: int, palette=None):
    if palette is None:
        if num_classes == 11:
            return DEFAULT_PALETTE_11
        rng = np.random.default_rng(0)
        return [tuple(rng.integers(0,255,size=3).tolist()) for _ in range(num_classes)]
    assert len(palette) >= num_classes
    return palette

def mask_to_color(mask: torch.Tensor, num_classes: int, palette=None) -> Image.Image:
    # mask: (H W) long
    palette = ensure_palette(num_classes, palette)
    mask_np = mask.cpu().numpy()
    h,w = mask_np.shape
    img = np.zeros((h,w,3), dtype=np.uint8)
    for cls in range(num_classes):
        img[mask_np==cls] = palette[cls]
    return Image.fromarray(img)

def save_color_mask(mask: torch.Tensor, path: str, num_classes: int, palette=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = mask_to_color(mask, num_classes, palette)
    img.save(path)
