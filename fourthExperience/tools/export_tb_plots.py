#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export training curves from CSV and (optionally) TensorBoard event files.

Generates PNG figures under ../figures/:
- training_loss.png: step vs train_loss and dev_loss
- dev_bleu.png: step vs dev_bleu
- learning_rate.png: step vs learning rate
- bleu_vs_time.png: time (sec) vs dev_bleu

Usage:
  python tools/export_tb_plots.py \
    --csv ../checkpoints/training_metrics.csv \
    --tb_dir ../checkpoints/nmt_experiment \
    --out_dir ../figures

TensorBoard is optional; if not installed, CSV-only plots are still produced.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def read_csv(csv_path: Path):
    import csv
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            for k in ["step", "epoch", "time_sec"]:
                if k in r and r[k] != "":
                    r[k] = int(float(r[k]))
            for k in ["train_loss", "dev_bleu", "dev_loss", "lr"]:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def plot_from_csv(rows, out_dir: Path):
    if not rows:
        print("[WARN] No rows found in CSV; skip CSV plots")
        return {}
    steps = [r["step"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    dev_loss = [r["dev_loss"] for r in rows]
    dev_bleu = [r["dev_bleu"] for r in rows]
    lrs = [r["lr"] for r in rows]
    times = [r["time_sec"] for r in rows]

    # figure 1: loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train_loss", color="#1f77b4")
    plt.plot(steps, dev_loss, label="dev_loss", color="#d62728")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Dev Loss vs Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_path = out_dir / "training_loss.png"
    plt.tight_layout(); plt.savefig(loss_path, dpi=200); plt.close()

    # figure 2: BLEU curve
    plt.figure(figsize=(8, 5))
    plt.plot(steps, dev_bleu, label="dev_bleu", color="#2ca02c")
    plt.xlabel("Step")
    plt.ylabel("BLEU")
    plt.title("Dev BLEU vs Step")
    plt.grid(True, alpha=0.3)
    bleu_path = out_dir / "dev_bleu.png"
    plt.tight_layout(); plt.savefig(bleu_path, dpi=200); plt.close()

    # figure 3: learning rate
    plt.figure(figsize=(8, 5))
    plt.plot(steps, lrs, label="lr", color="#9467bd")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule (Noam)")
    plt.grid(True, alpha=0.3)
    lr_path = out_dir / "learning_rate.png"
    plt.tight_layout(); plt.savefig(lr_path, dpi=200); plt.close()

    # figure 4: BLEU vs time
    if times and all(isinstance(t, int) for t in times):
        plt.figure(figsize=(8, 5))
        plt.plot(times, dev_bleu, label="dev_bleu", color="#ff7f0e")
        plt.xlabel("Time (sec)")
        plt.ylabel("BLEU")
        plt.title("Dev BLEU vs Time")
        plt.grid(True, alpha=0.3)
        bleu_time_path = out_dir / "bleu_vs_time.png"
        plt.tight_layout(); plt.savefig(bleu_time_path, dpi=200); plt.close()
    else:
        bleu_time_path = None

    best_bleu = max(dev_bleu) if dev_bleu else None
    best_row = None
    if best_bleu is not None:
        idx = dev_bleu.index(best_bleu)
        best_row = rows[idx]

    return {
        "loss_path": str(loss_path),
        "bleu_path": str(bleu_path),
        "lr_path": str(lr_path),
        "bleu_time_path": (str(bleu_time_path) if bleu_time_path else None),
        "best_bleu": best_bleu,
        "best_step": (best_row["step"] if best_row else None),
        "best_epoch": (best_row["epoch"] if best_row else None),
    }


def read_tb_scalars(tb_dir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("[INFO] TensorBoard not available; skip TB plots")
        return None
    if not tb_dir.exists():
        print(f"[INFO] TB dir not found: {tb_dir}")
        return None
    # Load with small size guidance to speed up
    acc = EventAccumulator(str(tb_dir), size_guidance={
        'scalars': 0,
        'images': 0,
        'histograms': 0,
        'tensors': 0,
        'compressedHistograms': 0,
    })
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    scalars = {tag: acc.Scalars(tag) for tag in tags}
    return scalars


def maybe_plot_tb_scalars(scalars, out_dir: Path):
    if not scalars:
        return {}
    out = {}
    for tag in [
        "train/loss", "train/lr", "dev/loss", "dev/bleu",
        "epoch/train_loss", "epoch/dev_loss", "epoch/dev_bleu",
    ]:
        if tag not in scalars:
            continue
        ev = scalars[tag]
        steps = [e.step for e in ev]
        vals = [e.value for e in ev]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, vals, label=tag)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.grid(True, alpha=0.3)
        p = out_dir / f"tb_{tag.replace('/', '_')}.png"
        plt.tight_layout(); plt.savefig(p, dpi=200); plt.close()
        out[tag] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Export training plots from CSV/TensorBoard")
    ap.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "training_metrics.csv"))
    ap.add_argument("--tb_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "nmt_experiment"))
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    tb_dir = Path(args.tb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if csv_path.exists():
        rows = read_csv(csv_path)
        summary.update(plot_from_csv(rows, out_dir))
    else:
        print(f"[WARN] CSV not found: {csv_path}")

    scalars = read_tb_scalars(tb_dir)
    tb_plots = maybe_plot_tb_scalars(scalars, out_dir)
    summary["tb_plots"] = tb_plots

    # Write a small JSON summary for the report to consume if needed
    import json
    with (out_dir / "plots_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] Plots exported to:", out_dir)
    if summary.get("best_bleu") is not None:
        print(f"Best Dev BLEU: {summary['best_bleu']:.2f} at step {summary.get('best_step')} (epoch {summary.get('best_epoch')})")


if __name__ == "__main__":
    main()
