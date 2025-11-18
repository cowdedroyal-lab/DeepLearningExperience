import torch
import numpy as np
from typing import Dict, Optional

# 计算语义分割指标：Pixel Acc, Mean Pixel Acc, Mean IoU, Confusion Matrix

def _fast_hist(true: torch.Tensor, pred: torch.Tensor, num_classes: int, ignore_index: int) -> np.ndarray:
    # true/pred: (H W) long
    mask = true != ignore_index
    true = true[mask].cpu().numpy().astype(np.int64)
    pred = pred[mask].cpu().numpy().astype(np.int64)
    hist = np.bincount(num_classes * true + pred, minlength=num_classes ** 2)
    return hist.reshape(num_classes, num_classes)

def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, ignore_index: int) -> np.ndarray:
    # 支持批次
    if y_true.dim() == 3:  # (B H W)
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            hist += _fast_hist(t, p, num_classes, ignore_index)
        return hist
    else:
        return _fast_hist(y_true, y_pred, num_classes, ignore_index)

def metrics_from_confusion(hist: np.ndarray) -> Dict[str, np.ndarray]:
    # hist: [C, C] 行是真实，列是预测
    tp = np.diag(hist)
    support = hist.sum(axis=1)
    pred_sum = hist.sum(axis=0)
    pixel_acc = tp.sum() / (hist.sum() + 1e-12)
    class_acc = tp / (support + 1e-12)
    mean_class_acc = np.mean(class_acc)
    iu = tp / (support + pred_sum - tp + 1e-12)
    mean_iu = np.mean(iu)
    return {
        'pixel_acc': pixel_acc,
        'mean_pixel_acc': mean_class_acc,
        'class_acc': class_acc,
        'class_iou': iu,
        'mean_iou': mean_iu,
    }

def evaluate_batch(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> Dict[str, np.ndarray]:
    # logits: (B,C,H,W) targets: (B,H,W)
    preds = torch.argmax(logits, dim=1)
    hist = compute_confusion_matrix(targets, preds, num_classes, ignore_index)
    return metrics_from_confusion(hist)

if __name__ == '__main__':
    # 随机自测
    B, C, H, W = 2, 5, 4, 4
    torch.manual_seed(0)
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    m = evaluate_batch(logits, targets, C, ignore_index=255)
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            print(k, v)
        else:
            print(k, v)
