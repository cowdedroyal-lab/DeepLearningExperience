# SegNet 街景分割工程 (CamVid)

本工程实现一个基于 SegNet 结构的街景语义分割模型，支持 CamVid 数据集（可扩展到 11 / 32 类），提供训练、验证、测试与推理脚本，以及像素精度、平均像素精度、Mean IoU 等评估指标。

## 目录结构规划
```
segnet/
  src/
    datasets/camvid.py        # 数据集定义与颜色映射
    models/segnet.py          # SegNet 模型实现 (含池化索引与 MaxUnpool)
    metrics/seg_metrics.py    # 指标与混淆矩阵
    utils/config.py           # 配置加载与校验
    utils/logger.py           # 日志与TensorBoard封装
    utils/visualize.py        # 预测结果着色/保存
  train.py                    # 训练脚本
  eval.py                     # 验证/测试脚本
  inference.py                # 单张/目录推理
  configs/camvid.yaml         # 默认配置
  requirements.txt            # 依赖
```

## 实验目的 (与指导书对应)
1. 掌握深度学习在计算机视觉中的应用——图像语义分割。
2. 通过实现与训练 SegNet，理解编码器-解码器结构、池化索引保存与还原机制。
3. 学习常用评估指标：Pixel Acc、Mean Pixel Acc、Mean IoU。

## SegNet 简介
SegNet 采用对称的 Encoder-Decoder 结构：
- Encoder: 多级 Conv + BN + ReLU + MaxPool(return_indices=True)
- Decoder: 使用保存的池化 indices 进行 MaxUnpool，再接 Conv + BN + ReLU 恢复空间分辨率。
- 最终通过 1x1 卷积映射到类别数 (C)，Softmax 或 CrossEntropyLoss 在训练阶段。

与 U-Net 不同，SegNet 不做 skip concat，而是仅使用池化位置索引恢复特征分布，提高内存利用效率。

## CamVid 数据集
- 分辨率常见 960x720 (可裁剪/缩放)。
- 标签是彩色 PNG，每种颜色对应一个语义类别。
- 可配置 11 类或 32 类映射（见 `camvid.py` 中 `DEFAULT_CLASS_COLORS_11` 等）。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 配置文件示例 (`configs/camvid.yaml`)
```yaml
experiment_name: segnet_camvid
seed: 42
num_classes: 11
ignore_index: 255
train_images_dir: /path/to/CamVid/train
train_labels_dir: /path/to/CamVid/train_labels
val_images_dir: /path/to/CamVid/val
val_labels_dir: /path/to/CamVid/val_labels
test_images_dir: /path/to/CamVid/test
test_labels_dir: /path/to/CamVid/test_labels
class_mode: camvid_11  # or camvid_32
input_height: 360
input_width: 480
batch_size: 8
epochs: 120
optimizer: adam
lr: 1e-3
weight_decay: 1e-4
lr_scheduler: cosine
warmup_epochs: 5
save_dir: ./outputs
log_interval: 50
val_interval: 1
amp: true
num_workers: 4
```

## 训练
```bash
# 32 类（camvid.yaml）：
python train.py --config configs/camvid.yaml
```

```bash
# 11 类（camvid11.yaml）：
python train.py --config configs/camvid11.yaml
```
可选参数：`--resume path/to/checkpoint.pt`，`--no-amp` 禁用混合精度。

## 验证 / 测试
```bash
# 32 类（camvid.yaml）：
python eval.py --config configs/camvid.yaml --checkpoint outputs/segnet_camvid/best.pt --split val
python eval.py --config configs/camvid.yaml --checkpoint outputs/segnet_camvid/best.pt --split test

# 11 类（camvid11.yaml）：
python eval.py --config configs/camvid11.yaml --checkpoint outputs/segnet_camvid11/best.pt --split val
python eval.py --config configs/camvid11.yaml --checkpoint outputs/segnet_camvid11/best.pt --split test
```

## 推理
```bash
# 32 类（输入可以是单图或目录，建议指向整理后的 images 目录）：
python inference.py --config configs/camvid.yaml --checkpoint outputs/segnet_camvid/best.pt \
  --input CamVid/converted/camvid32/test/images --output pred_vis/camvid32

# 11 类：
python inference.py --config configs/camvid11.yaml --checkpoint outputs/segnet_camvid11/best.pt \
  --input CamVid/converted/camvid11/test/images --output pred_vis/camvid11
```

提示：对于 32 类推理，脚本会读取 `configs/camvid.yaml` 中的 `class_colors_path` 来构建可视化调色板，使预测配色与标注一致；11 类默认采用内置 11 色调色板。

## 数据准备（已内置脚本）
使用 `tools/prepare_camvid_datasets.py` 根据官方分割清单将原始数据整理为标准结构（默认用软链接，节省磁盘）：
```bash
# 同时生成 32 类与 11 类两个数据集到 CamVid/converted 下
python tools/prepare_camvid_datasets.py --camvid-root CamVid --out-root CamVid/converted --link

# 只生成 32 类（或 11 类）
python tools/prepare_camvid_datasets.py --camvid-root CamVid --out-root CamVid/converted --sets 32 --link
python tools/prepare_camvid_datasets.py --camvid-root CamVid --out-root CamVid/converted --sets 11 --link

# 若需要实际拷贝文件（非软链接），用 --copy
python tools/prepare_camvid_datasets.py --camvid-root CamVid --out-root CamVid/converted --copy
```

## 指标说明
- Pixel Accuracy: 正确分类像素 / 总像素 (忽略 ignore_index)。
- Mean Pixel Accuracy: 对每类计算像素准确率再取平均。
- Mean IoU: IoU = TP / (TP + FP + FN)；对所有有效类平均。
- Confusion Matrix: 行是真实类，列是预测类。

## 工程特性
- 可复现：统一随机种子、`torch.backends.cudnn.deterministic`。
- 模块化：模型、数据、指标、工具分离。
- 日志：TensorBoard + 控制台进度条。
- 检查点：保存最新与 best (按 Mean IoU)。
- 混合精度：节省显存与加速。
- 支持类权重与 ignore_index。

## 后续可扩展
- 增加 Lovasz-Softmax / Focal Loss。
- 引入 OHEM (Online Hard Example Mining)。
- 模型替换：PSPNet / DeepLabV3+。

## 参考文献与仓库
- SegNet: Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Semantic Segmentation. https://arxiv.org/abs/1511.02680
- 原始指导书引用: https://arxiv.org/pdf/1511.00561
- Keras 实现参考结构与指标思路 (未直接复制代码)：`divamgupta/image-segmentation-keras`

> 说明：为避免版权风险，本实现仅借鉴公开论文结构与公开思路，所有 PyTorch 代码均重新编写。

## 许可证
本工程默认以 MIT 许可证发布 (可根据课程要求调整)。
