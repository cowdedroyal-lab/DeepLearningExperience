"""Compute class weights for CamVid (or similar) segmentation datasets.

支持多种常见策略（可参考 SegNet 论文中的 median frequency balancing 以及其它社区常用方案）：
1) median_freq  : median(freq_non_zero) / freq_i （原 SegNet 用法）
2) inverse_freq : 1 / freq_i
3) sqrt_inv     : 1 / sqrt(freq_i)
4) log_inv      : 1 / log(k + freq_i)  (k 默认 1.02, ENet 论文类似思路减少极端放大)
5) effective_num: 基于《Class-Balanced Loss Based on Effective Number of Samples》
                  w_i ∝ 1 - beta / (1 - beta^{n_i}), 我们最终取其倒数再归一化

工程增强：
 - 忽略 ignore_index 像素
 - 零样本类别可选择赋 0 或使用 max_cap
 - 归一化到均值=1 或和= num_classes
 - 输出 YAML / JSON 片段便于直接写入配置文件
 - 可指定最大/最小裁剪避免过大权重导致训练不稳定
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.datasets.camvid import CamVidDataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='训练集图像目录')
    ap.add_argument('--labels', required=True, help='训练集标签目录')
    ap.add_argument('--height', type=int, default=360, help='缩放高度（与训练保持一致）')
    ap.add_argument('--width', type=int, default=480, help='缩放宽度（与训练保持一致）')
    ap.add_argument('--num-classes', type=int, default=11, help='有效类别数（不含 ignore）')
    ap.add_argument('--ignore-index', type=int, default=255, help='忽略像素索引')
    ap.add_argument('--class-mode', type=str, default='camvid_11', choices=['camvid_11', 'camvid_32'])
    ap.add_argument('--class-colors-path', type=str, default=None, help='若使用彩色标签需提供颜色映射 JSON')

    # 方法与参数
    ap.add_argument('--method', type=str, default='median_freq', choices=[
        'median_freq', 'inverse_freq', 'sqrt_inv', 'log_inv', 'effective_num'
    ], help='权重计算策略')
    ap.add_argument('--beta', type=float, default=0.99, help='effective_num 方法的 beta')
    ap.add_argument('--log-k', type=float, default=1.02, help='log_inv 方法的平移常数 k')

    # 归一化/裁剪
    ap.add_argument('--normalize', type=str, default='mean', choices=['mean', 'sum', 'none'],
                    help='归一化方式：mean=均值归一(均值=1)，sum=和归一(和=num_classes)，none=不归一')
    ap.add_argument('--cap-high', type=float, default=50.0, help='权重最高裁剪值（防止极端放大）')
    ap.add_argument('--cap-low', type=float, default=0.01, help='权重最低裁剪值')
    ap.add_argument('--skip-zero', action='store_true', help='对于未出现的类权重置 0 而不是使用公式')

    # 输出
    ap.add_argument('--out-yaml', type=str, default=None, help='写出 YAML 片段文件（仅 class_weights 行）')
    ap.add_argument('--out-json', type=str, default=None, help='写出 JSON 文件: {"class_weights": [...]}')
    ap.add_argument('--precision', type=int, default=6, help='输出小数精度')
    return ap.parse_args()


def compute_weights(counts: np.ndarray, method: str, beta: float, log_k: float) -> np.ndarray:
    total = counts.sum()
    freq = counts / (total + 1e-12)
    nz_mask = freq > 0

    if method == 'median_freq':
        median = np.median(freq[nz_mask]) if nz_mask.any() else 0.0
        w = median / (freq + 1e-12)
    elif method == 'inverse_freq':
        w = 1.0 / (freq + 1e-12)
    elif method == 'sqrt_inv':
        w = 1.0 / (np.sqrt(freq) + 1e-12)
    elif method == 'log_inv':
        # ENet 常见： w_i = 1 / log(k + p_i)
        w = 1.0 / (np.log(log_k + freq) + 1e-12)
    elif method == 'effective_num':
        # Effective number: E_i = (1 - beta^{n_i})/(1 - beta)
        # 原论文建议使用倒数进行平衡: w_i ∝ (1 - beta)/(1 - beta^{n_i}), 我们取其倒数再归一避免过小类权重无限大
        E = (1 - np.power(beta, counts + 1e-12)) / (1 - beta + 1e-12)
        w = 1.0 / (E + 1e-12)
    else:
        raise ValueError(f'未知 method: {method}')
    return w, freq


def normalize_weights(w: np.ndarray, mode: str, num_classes: int) -> np.ndarray:
    if mode == 'mean':
        return w / (w.mean() + 1e-12)
    elif mode == 'sum':
        return w * (num_classes / (w.sum() + 1e-12))
    else:
        return w


def main():
    args = parse_args()
    ds = CamVidDataset(args.images, args.labels, args.height, args.width,
                       class_mode=args.class_mode, num_classes=args.num_classes,
                       ignore_index=args.ignore_index, class_colors_path=args.class_colors_path, augment=False)

    counts = np.zeros(args.num_classes, dtype=np.int64)
    for _, mask in tqdm(ds, desc='Counting pixels'):
        m = mask.numpy()
        # ignore_index 已经在数据集转换阶段处理；这里仅统计 0..num_classes-1
        for c in range(args.num_classes):
            counts[c] += (m == c).sum()

    weights_raw, freq = compute_weights(counts, args.method, args.beta, args.log_k)

    if args.skip_zero:
        weights_raw[freq == 0] = 0.0

    # 裁剪极值，防止训练不稳定
    weights_raw = np.clip(weights_raw, args.cap_low, args.cap_high)

    weights = normalize_weights(weights_raw, args.normalize, args.num_classes)

    # 结果四舍五入（仅用于文本输出，不影响内部精度）
    rounded = [round(float(x), args.precision) for x in weights.tolist()]

    print('Method              :', args.method)
    print('Counts per class    :', counts.tolist())
    print('Frequencies         :', [round(float(x), args.precision) for x in freq.tolist()])
    print('Raw weights         :', [round(float(x), args.precision) for x in weights_raw.tolist()])
    print('Normalized weights  :', rounded)
    print('\nYAML snippet (add to config yaml):')
    print(f'class_weights: {rounded}')

    if args.out_yaml:
        with open(args.out_yaml, 'w', encoding='utf-8') as fy:
            fy.write(f'class_weights: {rounded}\n')
        print(f'[写入] YAML 片段 -> {args.out_yaml}')
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as fj:
            json.dump({'class_weights': rounded, 'method': args.method, 'counts': counts.tolist(), 'freq': [float(x) for x in freq.tolist()]}, fj, ensure_ascii=False, indent=2)
        print(f'[写入] JSON -> {args.out_json}')


if __name__ == '__main__':
    main()
