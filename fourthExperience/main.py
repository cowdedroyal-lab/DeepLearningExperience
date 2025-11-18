#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust NMT training script on the NiuTrans sample-submission-version dataset.

Design goals:
- Engineering-stable: rely on torch.nn.Transformer (battle-tested), careful masks, label smoothing, Noam LR schedule.
- Strong defaults: sinusoidal positional encoding, embedding weight tying, Xavier init for >1D params.
- Sensible data handling: uses Dev-set if present, otherwise split a small dev from train; supports Test-set evaluation.
- Logging & checkpoints: CSV metrics and TensorBoard; save best checkpoint (with vocabs + config).
- Decoding: batched greedy with optional no-repeat-ngram to reduce repetitions.

This file is self-contained and runnable:
  python main.py --data-dir ./sample-submission-version --output-dir ./checkpoints

Author: Copilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False

# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Special tokens
# ----------------------------
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# ----------------------------
# Tokenization and I/O helpers
# ----------------------------

def tokenize(line: str) -> List[str]:
    # Data is already tokenized by whitespace; still .strip() to be safe
    return line.strip().split()


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


# ----------------------------
# Vocabulary
# ----------------------------
@dataclass
class VocabConfig:
    min_freq: int = 1
    max_size: Optional[int] = None  # None means unlimited except specials


class Vocab:
    def __init__(self, cfg: VocabConfig):
        self.cfg = cfg
        # stoi/itos: include specials in fixed order
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos: List[str] = list(SPECIAL_TOKENS)

    def build(self, corpus_tokens: Iterable[List[str]]):
        counter = Counter()
        for toks in corpus_tokens:
            counter.update(toks)
        # Apply thresholds and sort by (freq desc, token asc)
        items = [
            (tok, freq) for tok, freq in counter.items() if freq >= self.cfg.min_freq and tok not in self.stoi
        ]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.cfg.max_size is not None:
            items = items[: max(0, self.cfg.max_size - len(SPECIAL_TOKENS))]
        for tok, _ in items:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

    def to_dict(self) -> Dict:
        return {"cfg": asdict(self.cfg), "itos": self.itos}

    @staticmethod
    def from_dict(d: Dict) -> "Vocab":
        cfg = VocabConfig(**d.get("cfg", {}))
        v = Vocab(cfg)
        v.itos = list(d["itos"])  # type: ignore[index]
        v.stoi = {tok: i for i, tok in enumerate(v.itos)}
        return v


# ----------------------------
# Dataset & Collate
# ----------------------------
class NMTDataset(Dataset):
    """Keep raw tokens; collate will numericalize and pad.
    Each item is a dict with keys: src_tokens (List[str]), trg_tokens (List[str] or None for test)."""

    def __init__(self, src_lines: List[str], trg_lines: Optional[List[str]]):
        assert trg_lines is None or len(src_lines) == len(trg_lines)
        self.src_tok: List[List[str]] = [tokenize(x) for x in src_lines]
        self.trg_tok: Optional[List[List[str]]] = None
        if trg_lines is not None:
            self.trg_tok = [tokenize(x) for x in trg_lines]

    def __len__(self) -> int:
        return len(self.src_tok)

    def __getitem__(self, idx: int) -> Dict[str, Optional[List[str]]]:
        src = self.src_tok[idx]
        if self.trg_tok is None:
            return {"src": src, "trg": None}
        else:
            return {"src": src, "trg": self.trg_tok[idx]}


@dataclass
class CollateConfig:
    max_src_len: int = 150
    max_trg_len: int = 150
    add_sos_eos: bool = True


def pad_sequence(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class NMTCollate:
    def __init__(self, src_vocab: Vocab, trg_vocab: Vocab, cfg: CollateConfig):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.cfg = cfg

    def __call__(self, batch: List[Dict[str, Optional[List[str]]]]) -> Dict[str, torch.Tensor]:
        src_tok = []
        trg_tok = []
        for item in batch:
            s = item["src"] or []
            t = item["trg"]
            # truncate
            s = s[: self.cfg.max_src_len]
            src_tok.append(s)
            if t is not None:
                t = t[: self.cfg.max_trg_len]
                trg_tok.append(t)

        # numeric src with eos
        src_ids_list = []
        for s in src_tok:
            seq = self.src_vocab.encode(s)
            if self.cfg.add_sos_eos:
                seq = seq + [EOS_IDX]
            src_ids_list.append(seq)
        src_ids = pad_sequence(src_ids_list, PAD_IDX)
        src_key_padding_mask = src_ids.eq(PAD_IDX)  # (B,S)

        if len(trg_tok) == 0:
            return {
                "src_ids": src_ids,
                "src_key_padding_mask": src_key_padding_mask,
            }

        # numeric trg -> build input and output
        tgt_in_list = []  # starts with SOS
        tgt_out_list = []  # ends with EOS
        for t in trg_tok:
            t_ids = self.trg_vocab.encode(t)
            if self.cfg.add_sos_eos:
                tgt_in = [SOS_IDX] + t_ids
                tgt_out = t_ids + [EOS_IDX]
            else:
                tgt_in = t_ids
                tgt_out = t_ids
            tgt_in_list.append(tgt_in)
            tgt_out_list.append(tgt_out)

        tgt_in_ids = pad_sequence(tgt_in_list, PAD_IDX)
        tgt_out_ids = pad_sequence(tgt_out_list, PAD_IDX)
        tgt_key_padding_mask = tgt_in_ids.eq(PAD_IDX)

        return {
            "src_ids": src_ids,
            "tgt_in_ids": tgt_in_ids,
            "tgt_out_ids": tgt_out_ids,
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
        }


# ----------------------------
# Positional Encoding (sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ----------------------------
# Transformer model (nn.Transformer)
# ----------------------------
@dataclass
class ModelConfig:
    src_vocab_size: int
    trg_vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    share_embeddings: bool = True


class TransformerNMT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(cfg.trg_vocab_size, cfg.d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # use (B, S, D)
        )
        self.generator = nn.Linear(cfg.d_model, cfg.trg_vocab_size)
        # weight tying
        # Tie generator to target embedding regardless of src vocab size
        if cfg.share_embeddings:
            self.generator.weight = self.trg_embed.weight
        # init
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src_ids: (B,S)
        src = self.src_embed(src_ids) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory

    def decode(
        self,
        tgt_in_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.trg_embed(tgt_in_ids) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        # causal mask for target (T,T) True means mask
        T = tgt_in_ids.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), diagonal=1)
        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)  # (B,T,V)
        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_key_padding_mask)
        logits = self.decode(tgt_in_ids, memory, tgt_key_padding_mask, src_key_padding_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,  # (B,S)
        src_key_padding_mask: torch.Tensor,  # (B,S)
        max_len: int = 150,
        ngram_block: int = 0,  # 0 disables, otherwise block repeated n-grams (e.g., 3)
        eos_idx: int = EOS_IDX,
    ) -> torch.Tensor:
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)
        memory = self.encode(src_ids, src_key_padding_mask)
        ys = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        # for no-repeat-ngram, keep per-sample dict of history n-grams
        ngram_sets: Optional[List[set]] = None
        if ngram_block and ngram_block > 1:
            ngram_sets = [set() for _ in range(B)]
        for _ in range(max_len):
            tgt_key_padding_mask = ys.eq(PAD_IDX)
            logits = self.decode(ys, memory, tgt_key_padding_mask, src_key_padding_mask)
            next_token_logits = logits[:, -1, :]
            # avoid generating PAD and SOS after the first token
            next_token_logits[:, PAD_IDX] = -1e9
            next_token_logits[:, SOS_IDX] = -1e9
            # optional no-repeat-ngram: mask tokens that would create repeated n-grams
            if ngram_block and ngram_block > 1 and ngram_sets is not None:
                for i in range(B):
                    if finished[i]:
                        continue
                    if ys.size(1) >= ngram_block - 1:
                        prev = ys[i].tolist()[-(ngram_block - 1):]
                        for tok in range(next_token_logits.size(1)):
                            cand = tuple(prev + [tok])
                            if cand in ngram_sets[i]:
                                next_token_logits[i, tok] = -1e9
            next_ids = next_token_logits.argmax(dim=-1)  # (B,)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            # update finished and ngram sets
            for i in range(B):
                if not finished[i] and ys[i, -1].item() == eos_idx:
                    finished[i] = True
                if ngram_block and ngram_block > 1 and ngram_sets is not None:
                    if ys.size(1) >= ngram_block:
                        tup = tuple(ys[i, -ngram_block :].tolist())
                        ngram_sets[i].add(tup)
            if finished.all():
                break
        return ys  # includes SOS and potential EOS


# ----------------------------
# Noam LR Scheduler (per-step)
# ----------------------------
class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


# ----------------------------
# BLEU (corpus BLEU-4 with add-one smoothing); supports single or multi references
# ----------------------------

def corpus_bleu_4(hypotheses: List[List[str]], references: List[Sequence[List[str]] | List[str]]) -> float:
    # normalize references to List[List[str]] per hyp
    refs_norm: List[List[List[str]]] = []
    for ref in references:
        if len(ref) > 0 and isinstance(ref[0], str):  # type: ignore[index]
            refs_norm.append([list(ref)])  # type: ignore[list-item]
        else:
            refs_norm.append([list(r) for r in ref])  # type: ignore[union-attr]

    # Brevity penalty calculation uses reference with closest length to hypothesis
    hyp_len_sum = 0
    ref_len_sum = 0

    # n-gram precision with add-one smoothing
    precisions = []
    for n in range(1, 5):
        numer = 1
        denom = 1
        for hyp, ref_list in zip(hypotheses, refs_norm):
            hyp_ngrams = Counter([tuple(hyp[i : i + n]) for i in range(max(0, len(hyp) - n + 1))])
            max_ref_ngrams = defaultdict(int)
            # choose closest reference length for BP later
            # we compute this outside loop once per sentence; but it's cheap anyway
            # ngram clipping
            for ref in ref_list:
                ref_ngrams = Counter([tuple(ref[i : i + n]) for i in range(max(0, len(ref) - n + 1))])
                for k, v in ref_ngrams.items():
                    if v > max_ref_ngrams[k]:
                        max_ref_ngrams[k] = v
            # clip
            clipped = 0
            for k, v in hyp_ngrams.items():
                clipped += min(v, max_ref_ngrams.get(k, 0))
            numer += clipped
            denom += max(1, sum(hyp_ngrams.values()))
        precisions.append(numer / denom)

    # brevity penalty
    for hyp, ref_list in zip(hypotheses, refs_norm):
        hyp_len = len(hyp)
        hyp_len_sum += hyp_len
        # closest reference length
        closest_ref_len = min(ref_list, key=lambda r: (abs(len(r) - hyp_len), len(r)))
        ref_len_sum += len(closest_ref_len)
    if hyp_len_sum == 0:
        return 0.0
    if hyp_len_sum > ref_len_sum:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_sum / (hyp_len_sum + 1e-12))

    bleu = bp * math.exp(sum((1 / 4) * math.log(p + 1e-12) for p in precisions))
    return bleu * 100.0


# ----------------------------
# Data preparation for NiuTrans sample bundle
# ----------------------------
@dataclass
class DataPaths:
    data_dir: Path
    train_src: Path
    train_trg: Path
    dev_pairs: Optional[Path]  # Niu.dev.txt with "src ||| trg"
    test_src: Optional[Path]
    test_ref: Optional[Path]


def discover_dataset(root: Path) -> DataPaths:
    # Expected structure based on provided bundle
    tm = root / "TM-training-set"
    dev = root / "Dev-set" / "Niu.dev.txt"
    test_src = root / "Test-set" / "Niu.test.txt"
    test_ref = root / "Reference-for-evaluation" / "Niu.test.reference"

    train_src = tm / "chinese.txt"
    train_trg = tm / "english.txt"

    if not train_src.exists() or not train_trg.exists():
        raise FileNotFoundError(f"Training files not found under {tm}")

    return DataPaths(
        data_dir=root,
        train_src=train_src,
        train_trg=train_trg,
        dev_pairs=dev if dev.exists() else None,
        test_src=test_src if test_src.exists() else None,
        test_ref=test_ref if test_ref.exists() else None,
    )


@dataclass
class TrainConfig:
    # data & io
    data_dir: str
    output_dir: str
    run_name: str = "nmt_experiment"
    resume: Optional[str] = None
    # vocab
    src_min_freq: int = 1
    trg_min_freq: int = 1
    src_max_size: Optional[int] = None
    trg_max_size: Optional[int] = None
    # model
    d_model: int = 512
    nhead: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    ff_dim: int = 2048
    dropout: float = 0.1
    share_embeddings: bool = True
    # train
    batch_size: int = 128
    epochs: int = 20
    lr: float = 2.0  # base multiplier for Noam schedule
    warmup_steps: int = 4000
    weight_decay: float = 0.0
    label_smoothing: float = 0.1
    clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    # collate
    max_src_len: int = 150
    max_trg_len: int = 150
    # eval
    eval_every_steps: int = 2000
    max_decode_len: int = 150
    ngram_block: int = 3
    # early stopping
    patience: int = 5
    min_delta_bleu: float = 0.2
    # device
    force_cpu: bool = False
    # quick check
    sanity_check: bool = False
    inspect_data: bool = False


# ----------------------------
# Training / Evaluation helpers
# ----------------------------

def create_model_and_optimizer(cfg: TrainConfig, src_vocab: Vocab, trg_vocab: Vocab, device: torch.device):
    mcfg = ModelConfig(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
        share_embeddings=cfg.share_embeddings,
    )
    model = TransformerNMT(mcfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    return model, optimizer, scheduler


def build_loaders(cfg: TrainConfig, paths: DataPaths, device: torch.device):
    # read training data
    train_src_lines = read_lines(paths.train_src)
    train_trg_lines = read_lines(paths.train_trg)

    # dev data: prefer Dev-set if exists; otherwise split 2% or up to 2000 pairs as dev
    if paths.dev_pairs is not None:
        dev_pairs_lines = read_lines(paths.dev_pairs)
        dev_src_lines = []
        dev_trg_lines = []
        for ln in dev_pairs_lines:
            if "|||" in ln:
                s, t = ln.split("|||", 1)
                s = s.strip()
                t = t.strip()
                if s and t:
                    dev_src_lines.append(s)
                    dev_trg_lines.append(t)
        # Fallback: if Dev-set exists but contains no valid pairs, split from training
        if len(dev_src_lines) == 0:
            n = len(train_src_lines)
            k = min(max(1, int(0.02 * n)), 2000)
            idx = list(range(n))
            random.shuffle(idx)
            dev_idx = set(idx[:k])
            dev_src_lines = [train_src_lines[i] for i in sorted(dev_idx)]
            dev_trg_lines = [train_trg_lines[i] for i in sorted(dev_idx)]
            keep = [i for i in range(n) if i not in dev_idx]
            train_src_lines = [train_src_lines[i] for i in keep]
            train_trg_lines = [train_trg_lines[i] for i in keep]
            print(f"[WARN] Dev-set file found but had 0 valid pairs; fell back to split {k} examples from train")
    else:
        n = len(train_src_lines)
        k = min(max(1, int(0.02 * n)), 2000)
        idx = list(range(n))
        random.shuffle(idx)
        dev_idx = set(idx[:k])
        dev_src_lines = [train_src_lines[i] for i in sorted(dev_idx)]
        dev_trg_lines = [train_trg_lines[i] for i in sorted(dev_idx)]
        keep = [i for i in range(n) if i not in dev_idx]
        train_src_lines = [train_src_lines[i] for i in keep]
        train_trg_lines = [train_trg_lines[i] for i in keep]

    # Build vocabs from training only
    src_vocab = Vocab(VocabConfig(min_freq=cfg.src_min_freq, max_size=cfg.src_max_size))
    trg_vocab = Vocab(VocabConfig(min_freq=cfg.trg_min_freq, max_size=cfg.trg_max_size))
    src_vocab.build([tokenize(x) for x in train_src_lines])
    trg_vocab.build([tokenize(x) for x in train_trg_lines])

    # datasets
    train_ds = NMTDataset(train_src_lines, train_trg_lines)
    dev_ds = NMTDataset(dev_src_lines, dev_trg_lines)

    collate = NMTCollate(src_vocab, trg_vocab, CollateConfig(max_src_len=cfg.max_src_len, max_trg_len=cfg.max_trg_len))

    # loaders
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate,
    )

    # test (optional)
    test_loader = None
    test_refs_multi: Optional[List[List[List[str]]]] = None
    if paths.test_src is not None and paths.test_ref is not None:
        test_src_lines = read_lines(paths.test_src)
        test_ref_lines = read_lines(paths.test_ref)
        # Handle potential multi-reference file: e.g., refs = R * sources
        if len(test_ref_lines) % max(1, len(test_src_lines)) == 0 and len(test_ref_lines) > len(test_src_lines):
            R = len(test_ref_lines) // len(test_src_lines)
            print(f"[INFO] Detected Test multi-reference: {R} refs per source ({len(test_src_lines)} src, {len(test_ref_lines)} refs)")
            # Build multi-ref list grouped by source, assuming contiguous grouping per source
            test_refs_multi = []
            for i in range(len(test_src_lines)):
                refs_i = [tokenize(test_ref_lines[i * R + r]) for r in range(R)]
                test_refs_multi.append(refs_i)
            # For decoding, we don't need targets in dataset
            test_ds = NMTDataset(test_src_lines, None)
        else:
            # Fallback: 1-1 mapping (truncate to min for robustness)
            if len(test_src_lines) != len(test_ref_lines):
                m = min(len(test_src_lines), len(test_ref_lines))
                print(
                    f"[WARN] Test src/ref length mismatch: {len(test_src_lines)} vs {len(test_ref_lines)}; truncating to {m}"
                )
                test_src_lines = test_src_lines[:m]
                test_ref_lines = test_ref_lines[:m]
            test_ds = NMTDataset(test_src_lines, test_ref_lines)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            drop_last=False,
            collate_fn=collate,
        )

    return src_vocab, trg_vocab, train_loader, dev_loader, test_loader, test_refs_multi


def tokens_from_ids(ids: torch.Tensor, vocab: Vocab) -> List[str]:
    toks = []
    for i in ids.tolist():
        if i in (PAD_IDX, SOS_IDX):
            continue
        if i == EOS_IDX:
            break
        toks.append(vocab.itos[i])
    return toks


def _oov_rate(lines: List[str], vocab: Vocab) -> float:
    total = 0
    oov = 0
    for ln in lines:
        toks = tokenize(ln)
        total += len(toks)
        for t in toks:
            if t not in vocab.stoi:
                oov += 1
    return (oov / max(1, total)) * 100.0


@torch.no_grad()
def evaluate_bleu_and_loss(
    model: TransformerNMT,
    data_loader: DataLoader,
    trg_vocab: Vocab,
    device: torch.device,
    max_decode_len: int,
    ngram_block: int,
    label_smoothing: float,
    references_multi: Optional[List[List[List[str]]]] = None,
) -> Tuple[float, float]:
    model.eval()
    all_hyps: List[List[str]] = []
    all_refs: List[List[str]] = []
    losses: List[float] = []
    for batch in data_loader:
        src_ids = batch["src_ids"].to(device)
        src_kpm = batch["src_key_padding_mask"].to(device)
        # loss (if references available)
        if "tgt_in_ids" in batch:
            tgt_in = batch["tgt_in_ids"].to(device)
            tgt_out = batch["tgt_out_ids"].to(device)
            tgt_kpm = batch["tgt_key_padding_mask"].to(device)
            logits = model(src_ids, tgt_in, src_kpm, tgt_kpm)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1),
                ignore_index=PAD_IDX,
                label_smoothing=label_smoothing,
            )
            losses.append(loss.item())
        # decode
        ys = model.greedy_decode(src_ids, src_kpm, max_len=max_decode_len, ngram_block=ngram_block)
        # gather hyps/refs
        B = ys.size(0)
        for i in range(B):
            hyp_tok = tokens_from_ids(ys[i], trg_vocab)
            all_hyps.append(hyp_tok)
        if "tgt_out_ids" in batch:
            for tgt in batch["tgt_out_ids"]:
                ref_tok = tokens_from_ids(tgt, trg_vocab)
                all_refs.append(ref_tok)
    # Prefer external multi-references when dataset has no target refs
    if references_multi is not None and len(all_refs) == 0:
        bleu = corpus_bleu_4(all_hyps, references_multi)
    else:
        bleu = corpus_bleu_4(all_hyps, all_refs)
    loss_mean = float(np.mean(losses)) if losses else 0.0
    return bleu, loss_mean


# ----------------------------
# Checkpointing
# ----------------------------

def save_checkpoint(path: Path, model: TransformerNMT, src_vocab: Vocab, trg_vocab: Vocab, cfg: TrainConfig, metrics: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab": src_vocab.to_dict(),
            "trg_vocab": trg_vocab.to_dict(),
            "train_config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )


# ----------------------------
# Main train loop
# ----------------------------

def train_loop(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not cfg.force_cpu) else "cpu")
    paths = discover_dataset(Path(cfg.data_dir))

    src_vocab, trg_vocab, train_loader, dev_loader, test_loader, test_refs_multi = build_loaders(cfg, paths, device)

    model, optimizer, scheduler = create_model_and_optimizer(cfg, src_vocab, trg_vocab, device)

    # IO
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_metrics.csv"
    tb_writer = SummaryWriter(str(run_dir)) if TENSORBOARD_AVAILABLE else None

    # optional dataset inspection
    if cfg.inspect_data:
        train_ds: Dataset = train_loader.dataset  # type: ignore[assignment]
        dev_ds: Dataset = dev_loader.dataset  # type: ignore[assignment]
        print(f"[INFO] Train size: {len(train_ds)} | Dev size: {len(dev_ds)}")
        print(f"[INFO] Vocab sizes -> src: {len(src_vocab)} | trg: {len(trg_vocab)}")
        # Show a few samples
        def _show_sample(ds: Dataset, name: str):
            print(f"[INFO] Sample from {name}:")
            for i in range(min(2, len(ds))):
                ex = ds[i]
                print("SRC:", " ".join(ex["src"]))
                if ex["trg"] is not None:
                    print("TRG:", " ".join(ex["trg"]))
        _show_sample(train_ds, "train")
        _show_sample(dev_ds, "dev")
        # OOV on dev
        dev_src_lines = [" ".join(dev_loader.dataset[i]["src"]) for i in range(len(dev_loader.dataset))]  # type: ignore[index]
        dev_trg_lines = [" ".join(dev_loader.dataset[i]["trg"]) for i in range(len(dev_loader.dataset))]  # type: ignore[index]
        print(f"[INFO] Dev OOV src: {_oov_rate(dev_src_lines, src_vocab):.2f}% | trg: {_oov_rate(dev_trg_lines, trg_vocab):.2f}%")
        if test_loader is not None:
            print(f"[INFO] Test size: {len(test_loader.dataset)}")
            if test_refs_multi is not None and len(test_refs_multi) > 0:
                print(f"[INFO] Test multi-refs: {len(test_refs_multi[0])} per sample")
        return

    # optional sanity check
    if cfg.sanity_check:
        model.train()
        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_in = batch["tgt_in_ids"].to(device)
            tgt_out = batch["tgt_out_ids"].to(device)
            src_kpm = batch["src_key_padding_mask"].to(device)
            tgt_kpm = batch["tgt_key_padding_mask"].to(device)
            logits = model(src_ids, tgt_in, src_kpm, tgt_kpm)
            print("Sanity forward pass OK: logits shape", tuple(logits.shape))
            break
        print("Sanity check completed.")
        return

    # CSV header
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "train_loss", "dev_bleu", "dev_loss", "lr", "time_sec"])

    best_dev_bleu = -1.0
    best_ckpt_path = out_dir / "nmt_best.pt"
    steps = 0
    no_improve_epochs = 0

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            steps += 1
            src_ids = batch["src_ids"].to(device)
            tgt_in = batch["tgt_in_ids"].to(device)
            tgt_out = batch["tgt_out_ids"].to(device)
            src_kpm = batch["src_key_padding_mask"].to(device)
            tgt_kpm = batch["tgt_key_padding_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(src_ids, tgt_in, src_kpm, tgt_kpm)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1),
                ignore_index=PAD_IDX,
                label_smoothing=cfg.label_smoothing,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss.item(), steps)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)

            # periodic dev eval within epoch (optional)
            if cfg.eval_every_steps and (steps % cfg.eval_every_steps == 0):
                dev_bleu, dev_loss = evaluate_bleu_and_loss(
                    model, dev_loader, trg_vocab, device, cfg.max_decode_len, cfg.ngram_block, cfg.label_smoothing
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("dev/bleu", dev_bleu, steps)
                    tb_writer.add_scalar("dev/loss", dev_loss, steps)
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([steps, epoch, np.mean(epoch_losses), dev_bleu, dev_loss, scheduler.get_last_lr()[0], int(time.time() - start_time)])

        # end epoch: full dev evaluation
        dev_bleu, dev_loss = evaluate_bleu_and_loss(
            model, dev_loader, trg_vocab, device, cfg.max_decode_len, cfg.ngram_block, cfg.label_smoothing
        )
        train_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        if tb_writer is not None:
            tb_writer.add_scalar("epoch/train_loss", train_loss_mean, epoch)
            tb_writer.add_scalar("epoch/dev_bleu", dev_bleu, epoch)
            tb_writer.add_scalar("epoch/dev_loss", dev_loss, epoch)

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([steps, epoch, train_loss_mean, dev_bleu, dev_loss, scheduler.get_last_lr()[0], int(time.time() - start_time)])

        # early stopping + checkpointing on BLEU
        if dev_bleu > best_dev_bleu + cfg.min_delta_bleu:
            best_dev_bleu = dev_bleu
            no_improve_epochs = 0
            save_checkpoint(best_ckpt_path, model, src_vocab, trg_vocab, cfg, {"dev_bleu": dev_bleu, "epoch": epoch})
            print(f"[Epoch {epoch}] New best dev BLEU: {dev_bleu:.2f} -> saved {best_ckpt_path}")
        else:
            no_improve_epochs += 1
            print(f"[Epoch {epoch}] Dev BLEU {dev_bleu:.2f} (no improve count={no_improve_epochs})")
            if no_improve_epochs >= cfg.patience:
                print("Early stopping triggered.")
                break

    if tb_writer is not None:
        tb_writer.close()

    # optional test evaluation
    if test_loader is not None and best_dev_bleu >= 0:
        print("Evaluating on Test set with best model checkpoint...")
        # reload best
        state = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])  # type: ignore[index]
        model.to(device)
        test_bleu, test_loss = evaluate_bleu_and_loss(
            model, test_loader, trg_vocab, device, cfg.max_decode_len, cfg.ngram_block, cfg.label_smoothing, test_refs_multi
        )
        print(f"Test BLEU: {test_bleu:.2f} | Test loss: {test_loss:.4f}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Robust Transformer NMT Trainer (nn.Transformer-based)")
    # io
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "sample-submission-version"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--run-name", type=str, default="nmt_experiment")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--sanity-check", action="store_true", help="Run one forward pass and exit")
    p.add_argument("--inspect-data", action="store_true", help="Print dataset summary and a few samples, then exit")
    # vocab
    p.add_argument("--src-min-freq", type=int, default=1)
    p.add_argument("--trg-min-freq", type=int, default=1)
    p.add_argument("--src-max-size", type=int, default=0, help="0 for unlimited")
    p.add_argument("--trg-max-size", type=int, default=0, help="0 for unlimited")
    # model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--ff-dim", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--share-embeddings", action="store_true")
    # train
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    # collate
    p.add_argument("--max-src-len", type=int, default=150)
    p.add_argument("--max-trg-len", type=int, default=150)
    # eval
    p.add_argument("--eval-every-steps", type=int, default=2000)
    p.add_argument("--max-decode-len", type=int, default=150)
    p.add_argument("--ngram-block", type=int, default=3)
    # early stop
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta-bleu", type=float, default=0.2)

    args = p.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        src_min_freq=args.src_min_freq,
        trg_min_freq=args.trg_min_freq,
        src_max_size=(None if args.src_max_size == 0 else args.src_max_size),
        trg_max_size=(None if args.trg_max_size == 0 else args.trg_max_size),
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        share_embeddings=args.share_embeddings,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        clip_norm=args.clip_norm,
        seed=args.seed,
        num_workers=args.num_workers,
        max_src_len=args.max_src_len,
        max_trg_len=args.max_trg_len,
        eval_every_steps=args.eval_every_steps,
        max_decode_len=args.max_decode_len,
        ngram_block=args.ngram_block,
        patience=args.patience,
        min_delta_bleu=args.min_delta_bleu,
        force_cpu=args.force_cpu,
        sanity_check=args.sanity_check,
        inspect_data=args.inspect_data,
    )


def main():
    cfg = parse_args()
    train_loop(cfg)


if __name__ == "__main__":
    main()
