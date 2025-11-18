#!/usr/bin/env bash
set -euo pipefail

# 一键获取 YOLOv5 源码并安装依赖
# 注意：请先安装与你 CUDA 匹配的 PyTorch，否则可能会在 requirements 安装阶段失败。

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

echo "[INFO] Working dir: $ROOT_DIR"

if [ ! -d yolov5 ]; then
  echo "[INFO] Cloning ultralytics/yolov5 (tag v7.0) ..."
  git clone https://github.com/ultralytics/yolov5.git
  cd yolov5
  git checkout v7.0
else
  echo "[INFO] yolov5 directory already exists. Pulling latest and checking out v7.0 ..."
  cd yolov5
  git fetch --all
  git checkout v7.0
  git pull --ff-only || true
fi

echo "[INFO] Installing yolov5 requirements ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] YOLOv5 is ready at: $ROOT_DIR/yolov5"
