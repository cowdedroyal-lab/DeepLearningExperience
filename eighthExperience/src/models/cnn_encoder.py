"""
ResNet 图像特征抽取封装。

实现思路：
- 采用 torchvision.models.resnet50 预训练模型，移除 avgpool 和 fc，取最后卷积特征图 (B, 2048, H, W)。
- 展平空间维度得到 (B, L, 2048)，并通过线性层映射到 Transformer 的 d_model 维度。

参考/Copy 说明：
- CNN 特征抽取流程参考常规 image captioning 实践与 torchvision 官方 ResNet50 的层级结构拆解方法（仅用于教学示例）。
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
	def __init__(self, d_model: int = 512, pretrained: bool = True, fine_tune: bool = False):
		super().__init__()
		backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
		# 去掉 avgpool 和 fc，保留到 layer4 的输出（C=2048, H=7, W=7 for 224x224）
		self.cnn = nn.Sequential(*list(backbone.children())[:-2])

		c_out = 2048
		self.proj = nn.Linear(c_out, d_model)
		self.ln = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(p=0.1)

		# 冻结/微调策略
		for p in self.cnn.parameters():
			p.requires_grad = fine_tune

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		"""
		输入: images (B, 3, H, W)
		输出: feats (B, L, d_model), 其中 L=H'*W'
		"""
		with torch.set_grad_enabled(any(p.requires_grad for p in self.cnn.parameters())):
			conv = self.cnn(images)  # (B, C=2048, H', W')
		b, c, h, w = conv.shape
		feats = conv.flatten(2).transpose(1, 2)  # (B, L, C)
		feats = self.proj(feats)                 # (B, L, d_model)
		feats = self.ln(feats)
		feats = self.dropout(feats)
		return feats


__all__ = ["ResNetEncoder"]

