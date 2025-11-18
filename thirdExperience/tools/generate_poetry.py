#!/usr/bin/env python3
"""简洁的诗歌生成脚本。

加载 tang.npz 与 checkpoints/best_model.pth，给定起始字串生成一段续写。

示例：
  python tools/generate_poetry.py --start-words 湖光秋月两相和
  python tools/generate_poetry.py --start-words 春风又绿江南岸 --top-k 40 --temperature 0.9

参数说明：
  --start-words       起始字词（必需）
  --max-gen-len       最大生成长度（不含起始部分，默认 64）
  --top-k             Top-k 采样阈值，限制候选集合（默认 50）
  --temperature       温度，>1 更发散，<1 更保守（默认 1.0）
  --checkpoint        权重文件路径（默认 checkpoints/best_model.pth）
  --dataset-path      数据集 npz 路径（默认 tang.npz）
  --device            指定设备，默认自动检测
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def build_argparser():
    p = argparse.ArgumentParser(description='使用训练好的 LSTM 诗歌模型生成文本')
    p.add_argument('--start-words', type=str, required=True, help='起始字词')
    p.add_argument('--max-gen-len', type=int, default=64, help='最大生成长度')
    p.add_argument('--top-k', type=int, default=50, help='Top-k 阈值')
    p.add_argument('--temperature', type=float, default=1.0, help='温度系数')
    p.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='模型权重文件路径')
    p.add_argument('--dataset-path', type=str, default='tang.npz', help='词表与数据集 npz 文件')
    p.add_argument('--device', type=str, default=None, help='设备: cpu 或 cuda')
    return p


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5, bidirectional=False, weight_tying=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.weight_tying = weight_tying
        if weight_tying:
            self.proj = nn.Linear(hidden_dim * self.num_directions, embedding_dim)
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.linear = nn.Linear(hidden_dim * self.num_directions, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding_dropout(self.embeddings(x))
        out, hidden = self.lstm(emb, hidden)
        B, T, H = out.shape
        flat = out.reshape(B * T, H)
        if self.weight_tying:
            proj = self.proj(flat)
            logits = torch.matmul(proj, self.embeddings.weight.t()) + self.output_bias
        else:
            logits = self.linear(flat)
        return logits, hidden


def load_vocab(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    word2ix = data['word2ix'].item()
    ix2word = data['ix2word'].item()
    return word2ix, ix2word


@torch.no_grad()
def generate(model, start_words, ix2word, word2ix, device, max_gen_len, top_k, temperature):
    model.eval()
    results = list(start_words)
    start_idx = word2ix.get('<START>', 0)
    unk_idx = word2ix.get('<UNK>', start_idx)
    input_tensor = torch.tensor([start_idx], dtype=torch.long, device=device).view(1, 1)
    hidden = None
    for i in range(max_gen_len):
        logits, hidden = model(input_tensor, hidden)
        if i < len(start_words):
            ch = results[i]
            idx = word2ix.get(ch, unk_idx)
            input_tensor = torch.tensor([idx], device=device).view(1, 1)
            continue
        logits = logits.squeeze() / max(temperature, 1e-8)
        vocab_size = logits.size(-1)
        k = min(top_k, vocab_size)
        topk_logits, topk_indices = torch.topk(logits, k)
        probs = F.softmax(topk_logits, dim=-1)
        sel = torch.multinomial(probs, 1).item()
        token_idx = topk_indices[sel].item()
        token = ix2word[token_idx]
        if token == '<EOP>':
            break
        results.append(token)
        input_tensor = torch.tensor([token_idx], device=device).view(1, 1)
    return ''.join(results)


def main():
    args = build_argparser().parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    word2ix, ix2word = load_vocab(args.dataset_path)
    vocab_size = len(word2ix)

    # 与训练阶段保持相同的超参数（可按需修改）
    model = PoetryModel(vocab_size=vocab_size,
                        embedding_dim=256,
                        hidden_dim=512,
                        num_layers=2,
                        dropout=0.5,
                        bidirectional=False,
                        weight_tying=False)
    sd = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(sd)
    model.to(device)

    poem = generate(model,
                     start_words=args.start_words,
                     ix2word=ix2word,
                     word2ix=word2ix,
                     device=device,
                     max_gen_len=args.max_gen_len,
                     top_k=args.top_k,
                     temperature=args.temperature)
    print('=== 生成结果 ===')
    print(f'起始：{args.start_words}')
    print(poem)


if __name__ == '__main__':
    main()
