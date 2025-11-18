#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "yolov5"


def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cmd(cfg: dict, allow_missing: bool = False) -> list:
    py = sys.executable
    train_py = YOLO_DIR / "train.py"
    if not train_py.exists() and not allow_missing:
        raise FileNotFoundError(f"YOLOv5 train.py not found: {train_py}. Please run tools/fetch_yolov5.sh")

    args = [
        py, str(train_py),
        "--data", str(cfg.get("data", ROOT / "configs/dataset.yaml")),
        "--weights", str(cfg.get("weights", "yolov5s.pt")),
        "--imgsz", str(cfg.get("imgsz", 640)),
        "--epochs", str(cfg.get("epochs", 100)),
        "--batch", str(cfg.get("batch", 16)),
        "--project", str(cfg.get("project", ROOT / "runs/train")),
        "--name", str(cfg.get("name", "custom_yolov5s")),
    ]
    # optional
    if (dev := cfg.get("device", None)) is not None:
        args += ["--device", str(dev)]
    if cfg.get("rect", False):
        args += ["--rect"]
    if (workers := cfg.get("workers", None)) is not None:
        args += ["--workers", str(workers)]
    if (hyp := cfg.get("hyp", None)):
        args += ["--hyp", str(hyp)]
    return args


def main():
    ap = argparse.ArgumentParser(description="YOLOv5 train wrapper")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs/experiment.yaml"), help="path to experiment.yaml")
    ap.add_argument("--check", action="store_true", help="only print the command without running")
    args = ap.parse_args()

    cfg_p = Path(args.config).resolve()
    if not cfg_p.exists():
        print(f"[ERR] config not found: {cfg_p}")
        sys.exit(1)
    cfg = load_yaml(cfg_p)

    # sanity checks
    data_yaml = Path(cfg.get("data", ROOT / "configs/dataset.yaml")).resolve()
    if not data_yaml.exists():
        print(f"[ERR] dataset yaml not found: {data_yaml}")
        sys.exit(1)

    cmd = build_train_cmd(cfg, allow_missing=args.check)
    print("[INFO] Train command:\n  "+ shlex.join(cmd))
    if args.check:
        print("[CHECK] dry-run only, exit.")
        return

    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
