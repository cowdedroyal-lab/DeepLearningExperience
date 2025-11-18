#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "yolov5"


def main():
    ap = argparse.ArgumentParser(description="YOLOv5 eval wrapper (val.py)")
    ap.add_argument("--weights", type=str, required=True, help="path to .pt weights (e.g., runs/train/exp/weights/best.pt)")
    ap.add_argument("--data", type=str, default=str(ROOT / "configs/dataset.yaml"), help="dataset yaml path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--task", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    val_py = YOLO_DIR / "val.py"
    if not val_py.exists() and not args.check:
        print(f"[ERR] YOLOv5 val.py not found: {val_py}. Please run tools/fetch_yolov5.sh")
        sys.exit(1)

    cmd = [
        sys.executable, str(val_py),
        "--weights", args.weights,
        "--data", args.data,
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--device", str(args.device),
        "--task", args.task,
    ]

    print("[INFO] Eval command:\n  "+ shlex.join(cmd))
    if args.check:
        print("[CHECK] dry-run only, exit.")
        return

    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
