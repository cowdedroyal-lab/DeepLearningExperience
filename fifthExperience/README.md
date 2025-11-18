# 基于 YOLOv5 的目标检测实验（第五次实验）

本实验基于 PyTorch 与官方 YOLOv5 实现，完成自定义数据集的训练、推理与评估。我们采用官方教程与代码，外加本仓库提供的一组包装脚本，尽量做到“一条命令可复现”。

> 参考与致谢：Ultralytics YOLOv5 官方仓库与文档 https://github.com/ultralytics/yolov5

## 实验目标
- 理解 YOLOv5 的基本原理与工作流程（One-stage 检测器、锚框、损失等）。
- 掌握 YOLOv5 的部署与使用（数据准备、训练、推理、评估）。
- 在自定义数据集上训练，并使测试集 mAP@0.5 达到 90% 以上（数据与任务需可达到该标准）。

## 目录结构
```
fifthExperience/
  configs/
    dataset.yaml          # 数据集配置（YOLO 格式）
    experiment.yaml       # 训练超参与路径配置
  tools/
    fetch_yolov5.sh       # 一键获取 YOLOv5 源码并安装依赖（需网络）
    verify_yolo_dataset.py# 检查 YOLO 格式数据集合法性
  train.py                # 训练包装脚本 -> 调用 yolov5/train.py
  eval.py                 # 评估包装脚本 -> 调用 yolov5/val.py
  inference.py            # 推理包装脚本 -> 调用 yolov5/detect.py
  requirements.txt        # 本目录自用的轻量依赖（不含 torch 与 yolov5）
  yolov5/                 # 运行 tools/fetch_yolov5.sh 后生成
```

## 环境准备
1) 安装 Python 3.8+ 与 PyTorch（根据你的 CUDA 版本选择）：
- PyTorch 安装指南：https://pytorch.org/get-started/locally/

2) 安装本目录轻量依赖（仅用于包装脚本与数据检查脚本）：
```bash
pip install -r requirements.txt
```

3) 获取 YOLOv5 源码并安装其依赖（需可访问 GitHub）：
```bash
bash tools/fetch_yolov5.sh
```
- 脚本会克隆 yolov5 源码到 `fifthExperience/yolov5`，并安装其 requirements。
- 若你网络受限，可手工将官方仓库下载到该目录并手动执行 `pip install -r yolov5/requirements.txt`。

## 准备自定义数据集（YOLO 格式）
YOLO 数据集目录结构示例：
```
<dataset_root>/
  images/
    train/*.jpg|png|jpeg
    val/*.jpg|png|jpeg
    test/*.jpg|png|jpeg   # 可选
  labels/
    train/*.txt
    val/*.txt
    test/*.txt            # 可选
```
- 每个 `*.txt` 标签文件与同名图片一一对应，文件内部每一行形如：
  `class x_center y_center width height`（归一化到 [0,1]）。
- 在 `configs/dataset.yaml` 中填入 `train/val/test` 的 images 目录路径、类别数与类别名。

快速合法性检查：
```bash
python tools/verify_yolo_dataset.py --data configs/dataset.yaml
```

## 训练
你可以先 dry-run 查看即将执行的命令：
```bash
python train.py --check
```
正式训练（读取 `configs/experiment.yaml` 中的参数）：
```bash
python train.py
```
- 关键参数在 `configs/experiment.yaml`：
  - `data`: 数据集 YAML（默认 `configs/dataset.yaml`）
  - `weights`: 预训练权重（如 `yolov5s.pt`）
  - `imgsz`: 输入尺寸，常用 640
  - `epochs`, `batch`, `device`: 训练轮数、批大小、设备
  - `project`, `name`: 结果输出目录

训练产物：`runs/train/<name>`（由 YOLOv5 生成，包括曲线、混淆矩阵等）。

## 评估（验证/测试）
```bash
python eval.py \
  --weights runs/train/custom_yolov5s/weights/best.pt \
  --data configs/dataset.yaml
```
- 输出包含 `mAP@0.5` 与 `mAP@0.5:0.95` 等关键指标。
- 若你的数据与任务本身可达到 90%+ 的 mAP@0.5，可通过：更高分辨率、合适 batch、更多 epoch、数据增强策略等提升。

## 推理（单图/目录/视频）
```bash
python inference.py \
  --weights runs/train/custom_yolov5s/weights/best.pt \
  --source ./datasets/coco128/images/train2017/000000000030.jpg
```
- 结果保存在 `runs/detect/<name>` 目录。

## 常见问题与建议
- 若显存不足：降低 `batch` 或使用更小模型（`yolov5n/yolov5s`）。
- 精度未达标：
  - 确认数据标注质量与类别均衡；
  - 增加训练轮次 `epochs`；
  - 提升输入分辨率 `imgsz`；
  - 使用更大模型（`yolov5m/l/x`）与合适数据增强；
  - 冻结/解冻策略、余弦退火、EMA 等（参考官方教程）。
- 更完整参数说明与最佳实践：请查阅官方 README 与 wiki。

## 参考链接（官方）
- YOLOv5 仓库与 README：https://github.com/ultralytics/yolov5
- Train Custom Data 教程：https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
- 模型权重与配置：https://github.com/ultralytics/yolov5/releases

---
完成度说明：本目录提供“最小外壳包装 + 可复现脚本”，核心检测能力完全复用官方实现，便于稳健达成 >90% mAP@0.5 的目标。