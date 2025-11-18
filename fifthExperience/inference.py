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
    ap = argparse.ArgumentParser(description="YOLOv5 inference wrapper (detect.py)")
    ap.add_argument("--weights", type=str, required=True, help="path to .pt weights")
    ap.add_argument("--source", type=str, required=True, help="file/dir/URL/glob, e.g., data/images or 0 for webcam")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="exp")
    ap.add_argument("--save-txt", action="store_true")
    ap.add_argument("--save-conf", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    detect_py = YOLO_DIR / "detect.py"
    if not detect_py.exists() and not args.check:
        print(f"[ERR] YOLOv5 detect.py not found: {detect_py}. Please run tools/fetch_yolov5.sh")
        sys.exit(1)

    cmd = [
        sys.executable, str(detect_py),
        "--weights", args.weights,
        "--source", args.source,
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--device", str(args.device),
        "--project", args.project,
        "--name", args.name,
    ]
    if args.save_txt:
        cmd.append("--save-txt")
    if args.save_conf:
        cmd.append("--save-conf")

    print("[INFO] Detect command:\n  "+ shlex.join(cmd))
    if args.check:
        print("[CHECK] dry-run only, exit.")
        return

    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
