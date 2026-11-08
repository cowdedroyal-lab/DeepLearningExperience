# 深度学习课程实验合集
> 国科大深度学习课程的 8 次实验：从 MNIST 上的 CNN 一路做到目标检测、语义分割、图像描述与文本生成。

本仓库包含多个独立的深度学习实验项目，每个文件夹对应一次实验/一次大作业，实现了从经典卷积网络到目标检测、图像分割、图像描述与文本生成等多种任务。

## 问题

课程要求之外，这 8 次实验各自要回答的问题（它们是课程作业，下面只写仓库里能看到的做法）：

- 实验一：一个只有两层卷积的小 CNN，在 MNIST 上不加额外预处理能到什么水平。
- 实验二：中等规模的 ViT 在 32×32 的 CIFAR-10 上，靠数据增强和正则能做到多高的测试准确率。
- 实验三：字符级 LSTM 语言模型能否生成通顺的唐诗，Top-k 加温度采样对生成质量有什么影响。
- 实验四：从零用 `nn.Transformer` 做中译英，开发集 BLEU 能到多少。
- 实验五：用官方 YOLOv5 流程把训练、评估、推理串成一条命令可复现的流程。
- 实验六：SegNet 在 CamVid 上做 11 类和 32 类分割，难度差多少。
- 实验七：PTB 上的 LSTM 语言模型，复用官方示例，目标是困惑度低于 80。
- 实验八：ResNet50 加 Transformer 解码器做图像描述，并在 COCO 上评估。

## 方案

按实验列出（详见各子目录的 `README.md` 和 `实验报告.md`）：

- 实验一 `firstExperience/`：两层 5×5 卷积加最大池化、线性输出，交叉熵加 Adam，TensorBoard 记录，导出混淆矩阵与误分类样本图。
- 实验二 `secondExperience/`：ViT（4×4 patch、7 层、8 头、512 维），AdamW 加余弦退火，AutoAugment、随机裁剪、Dropout 0.2。
- 实验三 `thirdExperience/`：LSTM 语言模型，AdamW、ReduceLROnPlateau、梯度裁剪、提前停止，生成时用 Top-k 加温度采样。
- 实验四 `fourthExperience/`：`nn.Transformer` 序列到序列模型，标签平滑、Noam 学习率、带 n-gram 重复抑制的贪心解码，以开发集 BLEU 早停。
- 实验五 `fifthExperience/`：基于官方 YOLOv5 的训练、评估、推理包装脚本，附数据集合法性检查脚本。
- 实验六 `sixthExperience/`：SegNet（池化索引加 MaxUnpool，可选 VGG16-BN 预训练编码器），忽略 255 标签，可按类别频率加权；11 类和 32 类两套配置。
- 实验七 `seventhExperience/`：PTB 上的 LSTM 语言模型，复用 PyTorch 官方 word_language_model 示例，并标注了复制来源。
- 实验八 `eighthExperience/`：ResNet50 编码加 TransformerDecoder，COCO 2014，评估 BLEU、METEOR、ROUGE-L、CIDEr（简化实现）。

## 结果

已提交文本文件里能核对的数字（完整清单见 `.portfolio/project.yaml` 的 metrics）：

- 实验二：CIFAR-10 最佳测试准确率 90.82%（`secondExperience/实验报告.md`）。
- 实验四：开发集最佳 BLEU 23.50，第 30 轮（`fourthExperience/figures/plots_summary.json`）。
- 实验五：训练末轮 mAP@0.5 为 0.94657，但数据配置里 train 与 val 指向同一目录，不是留出集结果（`fifthExperience/runs/train/custom_yolov5s/results.csv`）。
- 实验六：CamVid-11 最佳 val mIoU 56.44%，CamVid-32 为 17.88%（`sixthExperience/实验报告.md`）。
- 实验一、三、七、八：仓库里没有可核对的最终数值（实验八的实验报告还是占位），这里不写数字。

## 复现

### 环境与依赖

不同实验的依赖略有差异，每个子目录下通常提供独立的 `requirements.txt`。建议为每个实验单独创建 Conda 或虚拟环境。例如：

```bash
cd eighthExperience
conda create -n eighthExperience python=3.10
conda activate eighthExperience
pip install -r requirements.txt
```

其它实验可参考对应子目录下的 `README.md` 或 `requirements.txt`。

### 使用说明

1. 选择感兴趣的实验子目录，例如 `fifthExperience/` 或 `eighthExperience/`；
2. 按该子目录下的 `README.md` 或 `实验报告.md` 中的说明准备数据和环境；
3. 运行训练脚本或推理脚本，例如：

   ```bash
   python train.py
   python eval.py
   python inference.py
   ```

## 证据清单

本仓库的证据化清单在 [`.portfolio/project.yaml`](.portfolio/project.yaml)：每条贡献、取舍、发现和指标都指向仓库里已提交的文件。校验命令：`profile-atlas check .`

## 目录结构概览

- `firstExperience/`：基于 MNIST 数据集的手写数字分类实验，包含简单 CNN 的训练、评估与可视化脚本。
- `secondExperience/`：在 CIFAR-10 数据集上使用 Vision Transformer (ViT) 的图像分类实验。
- `thirdExperience/`：中文古诗生成/语言建模实验，基于循环神经网络或类似结构实现诗歌自动生成。
- `fourthExperience/`：结构化数据/竞赛类任务（如表格数据预测）的实验代码与提交脚本。
- `fifthExperience/`：基于 YOLOv5 的目标检测实验，包含数据配置、训练、推理与结果可视化。
- `sixthExperience/`：在 CamVid 等数据集上的语义分割实验，包含训练、推理与分割可视化结果。
- `seventhExperience/`：循环或 Transformer 文本模型实验（如语言建模、文本生成），配有训练和推理脚本。
- `eighthExperience/`：图像描述（Image Captioning）实验，基于 COCO 数据集，实现图像特征提取与文字描述生成。

每个子目录中通常包含：

- `train.py` / `main.py`：模型训练入口脚本；
- `eval.py` / `inference.py`：模型评估或推理脚本；
- `configs/`：实验配置文件（如数据路径、模型超参数等）；
- `data/`：示例数据或数据组织结构；
- `runs/`、`checkpoints/`、`figures/` 等：日志、模型权重和可视化结果；
- `README.md`、`实验报告.md`：每次实验的具体说明和报告。

## 版权与致谢

本仓库仅用于课程实验与个人学习研究之用，部分代码参考或改写自公开开源项目（如 YOLOv5、常见图像分割与图像描述实现等）。如需在其它项目中使用，请根据各子项目中的 LICENSE 或原始开源仓库的协议进行引用与遵守。
