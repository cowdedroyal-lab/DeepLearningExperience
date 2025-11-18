# fourthExperience — 神经机器翻译 (NMT) 实验

这是一个针对神经机器翻译（NMT）实验的项目目录，包含预训练检查点、评估脚本和示例数据。该仓库适合作为复现实验、导出评估结果（例如 BLEU）以及查看训练过程日志的起点。

## 目录结构（摘要）

- `main.py` — 项目主入口（训练 / 推理 / 评估脚本，具体参数请查看文件内说明或运行 `-h`）。
- `checkpoints/` — 训练产生的模型与结果：
  - `nmt_best.pt` — 最佳模型（用于推理/评估）。
  - `predictions_bleu_sorted_reference.txt` — 已按bleu成绩排序的预测结果。
  - `training_metrics.csv` — 训练过程记录（指标随时间变化）。
  - `nmt_experiment/` — TensorBoard 事件文件（用于可视化训练日志）。
- `sample-submission-version/` — 示例数据集合，包含训练/开发/测试集的原始文件：
  - `Raw-data/`, `Dev-set/`, `Test-set/` 等 — 原始与划分好的句对文件。
- `Reference-for-evaluation/` — 用于评测的参考译文（例：`Niu.test.reference`）。
- `tools/` — 若干小工具脚本：
  - `export_bleu_sorted_reference.py` — 导出用于 BLEU 评估或生成排序的参考/预测对（请运行 `-h` 查看用法）。

## 目标与适用场景

- 复现或继续 NMT 训练/微调。
- 使用已有 `nmt_best.pt` 对测试集做推理并计算 BLEU。
- 生成可提交或可视化的评估结果（导出混淆、错误示例等）。

## 先决条件 / 依赖（建议）

以下是常见依赖，建议创建并安装：

- Python 3.8+
- PyTorch（根据你的 CUDA/CPU 环境选择对应版本）
- numpy
- sacrebleu 或 nltk（用于 BLEU 计算 / 标准化）
- tqdm

如果你使用 GPU，请安装对应 CUDA 版本的 `torch`，参考 PyTorch 官方安装页面。

## 快速开始 — 训练和评估

1. 开始训练：

```bash
python main.py
```

2. 开始评估：

```bash
# 在仓库根目录下运行（会打印帮助信息或执行默认行为）
python tools/export_bleu_sorted_reference.py
```

## 许可证 & 作者

```
MIT License

Copyright (c) 2025 cowdedroyal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```