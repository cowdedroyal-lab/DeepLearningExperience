#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_pairs(img_dir: Path, label_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs, labels = [], []
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() in IMG_EXTS:
            rel = img.relative_to(img_dir)
            lab = label_dir / rel.with_suffix(".txt")
            if lab.exists():
                imgs.append(img)
                labels.append(lab)
            else:
                print(f"[WARN] Missing label for: {img}")
    return imgs, labels


def check_label_file(label_p: Path, nc: int, max_lines: int = 3) -> Tuple[int, int, int]:
    ok = 0
    bad_fmt = 0
    bad_cls = 0
    lines = label_p.read_text(encoding="utf-8").strip().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            bad_fmt += 1
            if i < max_lines:
                print(f"  [BAD] {label_p}: line {i+1} expects 5 items, got {len(parts)} -> {line}")
            continue
        try:
            cls = int(float(parts[0]))
            xywh = list(map(float, parts[1:]))
        except ValueError:
            bad_fmt += 1
            if i < max_lines:
                print(f"  [BAD] {label_p}: non-numeric values -> {line}")
            continue
        if not (0 <= cls < nc):
            bad_cls += 1
            if i < max_lines:
                print(f"  [BAD] {label_p}: class index {cls} not in [0,{nc-1}]")
        x, y, w, h = xywh
        for k, val in zip(["x","y","w","h"],[x,y,w,h]):
            if not (0.0 <= val <= 1.0):
                bad_fmt += 1
                if i < max_lines:
                    print(f"  [BAD] {label_p}: {k}={val} not in [0,1]")
                break
        else:
            ok += 1
    return ok, bad_fmt, bad_cls


def main():
    ap = argparse.ArgumentParser(description="Verify YOLO-format dataset against dataset.yaml")
    ap.add_argument("--data", type=str, default="configs/dataset.yaml", help="path to dataset.yaml")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERR] dataset yaml not found: {data_yaml}")
        sys.exit(1)
    cfg = load_yaml(data_yaml)

    def _as_path(key: str):
        v = cfg.get(key, None)
        return Path(v).resolve() if v else None

    train = _as_path("train")
    val = _as_path("val")
    test = _as_path("test")
    nc = int(cfg.get("nc", -1))
    names = cfg.get("names", [])

    if nc <= 0:
        print("[ERR] nc must be > 0 in dataset.yaml")
        sys.exit(1)
    if not isinstance(names, list) or len(names) != nc:
        print(f"[WARN] names length ({len(names)}) != nc ({nc}). Please ensure they match.")

    def check_split(split_images: Path, split: str):
        if split_images is None:
            print(f"[INFO] {split}: skipped")
            return
        if not split_images.exists():
            print(f"[ERR] {split} images path not found: {split_images}")
            sys.exit(1)
        # infer labels dir: use images subdirectory name (e.g., train, train2017, val, val2017)
        label_subdir = split_images.name  # directory under images/
        labels_dir = split_images.parents[1] / "labels" / label_subdir
        if not labels_dir.exists():
            # fallback to legacy pattern labels/<split>
            legacy = split_images.parents[1] / "labels" / split
            if legacy.exists():
                labels_dir = legacy
            else:
                print(f"[ERR] {split} labels path not found: {labels_dir} or {legacy}")
                sys.exit(1)
        imgs, labs = find_pairs(split_images, labels_dir)
        print(f"[INFO] {split}: images={len(imgs)}, labeled={len(labs)}")
        # sample check first 100 labels
        from itertools import islice
        total_ok = total_bf = total_bc = 0
        for lp in islice(labs, 0, 100):
            ok, bf, bc = check_label_file(lp, nc)
            total_ok += ok
            total_bf += bf
            total_bc += bc
        if total_bf or total_bc:
            print(f"[WARN] {split}: bad_format={total_bf}, bad_class={total_bc} (checked up to 100 files)")
        else:
            print(f"[OK] {split}: label format looks good (checked up to 100 files)")

    check_split(train, "train")
    check_split(val, "val")
    check_split(test, "test")

    print("[DONE] Dataset verification finished.")


if __name__ == "__main__":
    main()
