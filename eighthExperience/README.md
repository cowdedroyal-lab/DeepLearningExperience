# 实验 8：基于 CNN-Transformer 的图像描述

本工程实现了一个基于 ResNet50 + TransformerDecoder 的图像描述系统，支持 MSCOCO 数据集的数据准备、训练、评估（BLEU/METEOR/ROUGE-L/CIDEr，简化实现）和单图推理。

实现说明：
- CNN 特征抽取：参考常规 image captioning 实践与 torchvision ResNet50 官方实现（仅用于教学示例）。
- Transformer 位置编码与解码器掩码：Copy/Adapted from PyTorch Transformer 教程（教学用途）。
- 数据组织/三分：样本级（每条 caption 一条样本）的构造与从 val2014 拆分 val/test 的思路参考了 Meshed-Memory Transformer（CVPR’20）仓库的 COCO 处理逻辑，但本仓库为从零实现的简化版本，便于课程阅读与维护。

目录结构（关键项）：

```
configs/                 # 配置
data/coco/               # COCO 数据根目录（images+annotations）
src/
	datasets/              # COCO Dataset 与 BertTokenizer 封装
	models/                # ResNetEncoder 与 CaptionTransformer
	utils/                 # 通用工具与评测指标（简化实现）
tools/
	download_coco.py       # 官方链接下载/解压 COCO 2014 数据
	train.py               # 训练入口
	eval.py                # 评估入口（val/test）
	inference.py           # 单图推理
```

## 1. 环境依赖

建议 Python 3.9+，需要安装下列依赖：

```
torch
torchvision
transformers
pycocotools
Pillow
PyYAML
tqdm
requests
nltk   # BLEU/METEOR（若 METEOR 缺少 wordnet，会自动降级）
```

> 注：根据你的 CUDA 版本安装匹配的 `torch/torchvision`，详见 PyTorch 官网说明。

## 2. 数据准备（MSCOCO 2014）

- 若你本地已有 COCO 2014 数据，确保目录为：

```
data/coco/
	annotations/
		captions_train2014.json
		captions_val2014.json
	train2014/
	val2014/
```

- 否则可执行下载脚本（耗时且占用空间大）：

```bash
python tools/download_coco.py --out_dir data/coco
```

配置文件 `configs/coco_baseline.yaml` 中默认路径与该结构一致。

## 3. 训练

```bash
python tools/train.py \
	--config configs/coco_baseline.yaml \
	--epochs 10 \
	--batch_size 64 \
	--lr 1e-4 \
	--device cuda
```

说明：
- 模型：ResNet50 编码 + Transformer 解码（d_model=512, nhead=8, layers=6）。
- 优化器：Adam(lr=1e-4, betas=(0.9, 0.98))；学习率调度：StepLR（可在脚本参数调整）。
- Loss：CrossEntropy(ignore_index=pad)。
- 评估：每 epoch 结束计算 val loss + BLEU（1~4），并按 BLEU-4 保存最佳模型到 `checkpoints/coco_caption_best.pt`。

## 4. 评估（val/test）

```bash
python tools/eval.py \
	--config configs/coco_baseline.yaml \
	--checkpoint checkpoints/coco_caption_best.pt \
	--split test \
	--batch_size 64 \
	--device cuda
```

输出一个字典，包含 BLEU-1/4、METEOR、ROUGE-L、CIDEr（CIDEr 为简化实现，仅用于教学评估）。

> 若 `nltk` 的 METEOR 因资源缺失报错，可在联网环境下执行一次：
> ```bash
> python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
> ```
> 下载失败则自动降级为基于重叠的 F1。

## 5. 单图推理

```bash
python tools/inference.py \
	--config configs/coco_baseline.yaml \
	--checkpoint checkpoints/coco_caption_best.pt \
	--image data/coco/val2014/COCO_val2014_000000000073.jpg \
	--device cuda
```

终端将打印生成的字幕。可通过 `--output path.txt` 将结果写入文件。

## 6. 复制/参考说明（用于报告）

- CNN 特征抽取：参考 torchvision ResNet50 官方模型的常规拆解与 image captioning 实践（教学用途）。
- 位置编码与自回归掩码：Copy/Adapted from PyTorch 官方 Transformer 教程（教学用途）。
- COCO 样本组织与 val/test 拆分的思路：参考 aimagelab/meshed-memory-transformer（CVPR’20）仓库的数据处理策略；本项目为从零实现的简化版本，未直接复用其复杂容器/模块。

## 7. 常见问题

- ImportError: transformers / pycocotools / nltk 未安装：请按“环境依赖”章节安装。
- CUDA/torch 安装问题：请按 PyTorch 官网指引安装与你显卡驱动匹配的版本。
- 评估分数偏低：默认仅 greedy 生成，可在后续加入 beam search；或增大模型/训练轮数/学习率调度策略。

