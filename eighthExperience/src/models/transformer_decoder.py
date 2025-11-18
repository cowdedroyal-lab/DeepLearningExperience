"""
Transformer 解码器 & 整体 CaptionModel

两种实现路线：
1) 直接复用 aimagelab/meshed-memory-transformer 的 Transformer + Decoder（功能强大但模块较多）。
2) 使用 PyTorch 内置 nn.TransformerDecoder，结合我们自己的 ResNetEncoder（更轻量易懂）。

本文件选择路线(2)，以确保结构清晰和可维护；同时在掩码构造与位置编码实现上参考了 PyTorch 官方教程。

Copy/参考说明：
- PositionalEncoding 与 subsequent mask 的写法参考自 PyTorch 官方 Transformer 教程（仅用于教学）。
- 生成（greedy）流程与 beam search 的思路参考了 M² Transformer 的评测代码，但此处实现简化版 greedy 以保证稳定性。
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

from src.models.cnn_encoder import ResNetEncoder


class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding (Copy/Adapted from PyTorch tutorial)."""

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: (seq_len, batch_size, d_model)
		"""
		x = x + self.pe[: x.size(0)]
		return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
	"""Generate a square mask for the sequence. (Copy/Adapted from PyTorch tutorial)
	形状: (sz, sz)，上三角为 -inf 以屏蔽后续位置。
	"""
	mask = torch.full((sz, sz), float('-inf'), device=device)
	mask = torch.triu(mask, diagonal=1)
	return mask


class CaptionTransformer(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		d_model: int = 512,
		nhead: int = 8,
		num_decoder_layers: int = 6,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		max_len: int = 64,
		pad_idx: int = 0,
		bos_id: int = 101,
		eos_id: int = 102,
		fine_tune_cnn: bool = True,
	):
		super().__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.pad_idx = pad_idx
		self.bos_id = bos_id
		self.eos_id = eos_id

		# Visual encoder
		self.visual_encoder = ResNetEncoder(d_model=d_model, pretrained=True, fine_tune=fine_tune_cnn)
		self.pos_enc_vis = PositionalEncoding(d_model, dropout)

		# Text embedding + positional encoding
		self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
		self.pos_enc_txt = PositionalEncoding(d_model, dropout)

		# Transformer decoder
		decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
												   dropout=dropout, batch_first=False)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
		self.norm = nn.LayerNorm(d_model)
		self.generator = nn.Linear(d_model, vocab_size)

		self._reset_parameters()

	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
		"""
		返回 memory: (L, B, d_model)
		"""
		feats = self.visual_encoder(images)  # (B, L, d)
		memory = feats.transpose(0, 1)       # (L, B, d)
		memory = self.pos_enc_vis(memory)
		return memory

	def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
		"""
		训练前向：teacher forcing
		images: (B, 3, H, W)
		captions: (B, T) 例如 [CLS] ... [SEP] [PAD] ...
		返回: logits (B, T-1, vocab_size)，对应输入 captions 的右移目标（排除最后一个 token）。
		"""
		device = images.device
		memory = self.encode_visual(images)  # (L, B, d)

		tgt_inp = captions[:, :-1]  # (B, T-1)
		tgt = self.tok_embed(tgt_inp).transpose(0, 1)  # (T-1, B, d)
		tgt = self.pos_enc_txt(tgt)

		T_in = tgt.size(0)
		tgt_mask = generate_square_subsequent_mask(T_in, device=device)  # (T-1, T-1)
		tgt_key_padding_mask = (tgt_inp == self.pad_idx)  # (B, T-1)

		dec_out = self.decoder(
			tgt,
			memory,
			tgt_mask=tgt_mask,
			tgt_key_padding_mask=tgt_key_padding_mask,
			memory_key_padding_mask=None,
		)  # (T-1, B, d)
		dec_out = self.norm(dec_out)
		logits = self.generator(dec_out.transpose(0, 1))  # (B, T-1, vocab)
		return logits

	@torch.no_grad()
	def generate(self, images: torch.Tensor, max_len: int = 30, greedy: bool = True) -> torch.Tensor:
		"""
		贪心生成：返回 token ids 序列 (B, T_gen)。
		"""
		device = images.device
		memory = self.encode_visual(images)  # (L, B, d)
		B = images.size(0)

		ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)  # 初始 [CLS]

		finished = torch.zeros(B, dtype=torch.bool, device=device)
		for _ in range(max_len):
			tgt = self.tok_embed(ys).transpose(0, 1)  # (t, B, d)
			tgt = self.pos_enc_txt(tgt)
			T_cur = tgt.size(0)
			tgt_mask = generate_square_subsequent_mask(T_cur, device=device)
			tgt_key_padding_mask = (ys == self.pad_idx)  # (B, t)
			dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
			dec_out = self.norm(dec_out)
			logits = self.generator(dec_out[-1])  # (B, vocab)
			next_token = torch.argmax(logits, dim=-1)
			ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

			finished |= (next_token == self.eos_id)
			if torch.all(finished):
				break

		return ys


__all__ = [
	"CaptionTransformer",
	"PositionalEncoding",
]

