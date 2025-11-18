#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export per-sentence BLEU (sorted desc) for Niu.test.reference using the
best Transformer checkpoint produced by `main.py` (nmt_best.pt).

Workflow:
1. Load checkpoint `checkpoints/nmt_best.pt` (created by `main.py`).
2. Restore source/target vocab and training config.
3. Parse Chinese/English pairs from `sample-submission-version/Reference-for-evaluation/Niu.test.reference`.
4. Batch greedy decode Chinese sentences with the trained model (respecting ngram_block + max_decode_len).
5. Compute sentence-level BLEU using the same `corpus_bleu_4` implementation from `main.py` (call with single hyp/ref).
6. Sort results by BLEU descending and write to `checkpoints/predictions_bleu_sorted_reference.txt`.

The output format:
Corpus BLEU: <float>
Total sentences: <int>

=== Predictions ranked by BLEU (desc) ===
[BLEU xx.xx]
CN  : <source Chinese tokens>
REF : <reference English tokens>
PRED: <predicted English tokens>
---

This script purposefully only depends on public objects from `main.py` to ensure
consistency with training implementation.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import List, Tuple

import torch

# Ensure parent directory (package root) on sys.path so we can import main.py
import sys as _sys
_pkg_root = Path(__file__).resolve().parents[1]  # fourthExperience root
if str(_pkg_root) not in _sys.path:
    _sys.path.insert(0, str(_pkg_root))

# Import components from training script main.py (same directory as package root)
from main import (  # type: ignore
    Vocab,
    TransformerNMT,
    corpus_bleu_4,
    tokens_from_ids,
    PAD_IDX,
    EOS_IDX,
)


REFERENCE_REL_PATH = "sample-submission-version/Reference-for-evaluation/Niu.test.reference"
CHECKPOINT_PATH = "checkpoints/nmt_best.pt"
OUTPUT_PATH = "checkpoints/predictions_bleu_sorted_reference.txt"


def is_chinese_line(s: str) -> bool:
    """Detect if a line contains any CJK character."""
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def is_english_line(s: str) -> bool:
    """Heuristic: contains Latin letters and no CJK."""
    has_ascii = bool(re.search(r"[A-Za-z]", s))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", s))
    return has_ascii and not has_cjk


def load_cn_en_pairs(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Parse Niu.test.reference file. We assume pattern Chinese line followed by English line.
    Robustness: skip empty lines; for each Chinese line, seek the next English line.
    Returns tokenized (whitespace split) lists.
    """
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    cn_tok: List[List[str]] = []
    en_tok: List[List[str]] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if is_chinese_line(line):
            # look ahead for first English line
            j = i + 1
            while j < len(raw_lines) and not is_english_line(raw_lines[j]):
                j += 1
            if j < len(raw_lines):
                cn_tok.append(line.split())
                en_tok.append(raw_lines[j].split())
                i = j + 1
            else:
                # No matching English line; discard this Chinese line
                i += 1
        else:
            i += 1
    return cn_tok, en_tok


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[TransformerNMT, Vocab, Vocab]:
    """Load model + vocabs from checkpoint saved by main.py."""
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in state:
        raise RuntimeError("Checkpoint missing 'model_state' key; ensure this comes from main.py training.")
    src_vocab = Vocab.from_dict(state["src_vocab"])  # type: ignore[arg-type]
    trg_vocab = Vocab.from_dict(state["trg_vocab"])  # type: ignore[arg-type]
    train_cfg = state.get("train_config", {})

    # Recreate model config consistent with training
    from main import ModelConfig  # local import matching earlier import fix
    mcfg = ModelConfig(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        d_model=train_cfg.get("d_model", 512),
        nhead=train_cfg.get("nhead", 8),
        num_encoder_layers=train_cfg.get("enc_layers", 6),
        num_decoder_layers=train_cfg.get("dec_layers", 6),
        dim_feedforward=train_cfg.get("ff_dim", 2048),
        dropout=train_cfg.get("dropout", 0.1),
        share_embeddings=train_cfg.get("share_embeddings", True),
    )
    model = TransformerNMT(mcfg).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, src_vocab, trg_vocab


def batch_encode_sources(src_sent_tokens: List[List[str]], src_vocab: Vocab, add_eos: bool = True, batch_size: int = 128) -> List[torch.Tensor]:
    """Encode source token lists into batches of padded id tensors (B,S)."""
    batches: List[torch.Tensor] = []
    for i in range(0, len(src_sent_tokens), batch_size):
        chunk = src_sent_tokens[i:i + batch_size]
        encoded = []
        for toks in chunk:
            ids = src_vocab.encode(toks)
            if add_eos:
                ids = ids + [EOS_IDX]
            encoded.append(ids)
        # pad
        max_len = max(len(x) for x in encoded)
        padded = []
        for seq in encoded:
            padded.append(seq + [PAD_IDX] * (max_len - len(seq)))
        batch_tensor = torch.tensor(padded, dtype=torch.long)
        batches.append(batch_tensor)
    return batches


def make_padding_mask(batch_ids: torch.Tensor) -> torch.Tensor:
    """Return key_padding_mask (B,S) where True indicates padding positions."""
    return batch_ids.eq(PAD_IDX)


def decode_batches(model: TransformerNMT, batches: List[torch.Tensor], device: torch.device,
                   max_len: int, ngram_block: int, trg_vocab: Vocab) -> List[List[str]]:
    all_preds: List[List[str]] = []
    for batch_ids in batches:
        batch_ids = batch_ids.to(device)
        kpm = make_padding_mask(batch_ids).to(device)
        with torch.no_grad():
            out_ids = model.greedy_decode(batch_ids, kpm, max_len=max_len, ngram_block=ngram_block)
        for seq_ids in out_ids:
            toks = tokens_from_ids(seq_ids, trg_vocab)
            all_preds.append(toks)
    return all_preds


def sentence_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute sentence-level BLEU using corpus_bleu_4 on single pair (returns percentage)."""
    return corpus_bleu_4([pred_tokens], [ref_tokens])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_path = Path(REFERENCE_REL_PATH)
    ckpt_path = Path(CHECKPOINT_PATH)
    out_path = Path(OUTPUT_PATH)

    print(f"[INFO] Device: {device}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (run main.py training first)")

    print("[INFO] Loading checkpoint and rebuilding model/vocabs...")
    model, src_vocab, trg_vocab = build_model_from_checkpoint(ckpt_path, device)

    # training config values (for decode hyperparams)
    ckpt_state = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt_state.get("train_config", {})
    max_decode_len = int(train_cfg.get("max_decode_len", 150))
    ngram_block = int(train_cfg.get("ngram_block", 3))

    print("[INFO] Parsing reference file for CN-EN pairs...")
    cn_tok, en_tok = load_cn_en_pairs(ref_path)
    if not cn_tok:
        raise RuntimeError("Parsed zero sentence pairs; check reference file format.")
    print(f"[INFO] Parsed {len(cn_tok)} sentence pairs.")

    print("[INFO] Encoding sources into batches...")
    batches = batch_encode_sources(cn_tok, src_vocab, add_eos=True, batch_size=128)

    print("[INFO] Decoding predictions with greedy search (ngram_block=%d, max_len=%d)..." % (ngram_block, max_decode_len))
    pred_tok = decode_batches(model, batches, device, max_len=max_decode_len, ngram_block=ngram_block, trg_vocab=trg_vocab)

    print("[INFO] Computing sentence BLEU scores...")
    entries = []
    for idx, (src_tokens, ref_tokens, hyp_tokens) in enumerate(zip(cn_tok, en_tok, pred_tok)):
        b = sentence_bleu(hyp_tokens, ref_tokens)  # percentage already
        entries.append({
            "idx": idx,
            "bleu": b,
            "cn": " ".join(src_tokens),
            "ref": " ".join(ref_tokens),
            "pred": " ".join(hyp_tokens),
        })

    # Corpus BLEU for information
    corpus_bleu = corpus_bleu_4(pred_tok, en_tok)

    print("[INFO] Sorting by BLEU descending and writing output...")
    entries.sort(key=lambda x: x["bleu"], reverse=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Corpus BLEU: {corpus_bleu:.2f}\n")
        f.write(f"Total sentences: {len(entries)}\n")
        f.write("\n=== Predictions ranked by BLEU (desc) ===\n")
        for e in entries:
            f.write(f"[BLEU {e['bleu']:.2f}]\n")
            f.write(f"CN  : {e['cn']}\n")
            f.write(f"REF : {e['ref']}\n")
            f.write(f"PRED: {e['pred']}\n")
            f.write("---\n")
    print(f"[INFO] Done. Output written to: {out_path}")


if __name__ == "__main__":
    main()
