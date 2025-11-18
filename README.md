# 深度学习课程实验项目汇总

本仓库包含多个独立的深度学习实验项目，每个文件夹对应一次实验/一次大作业，实现了从经典卷积网络到目标检测、图像分割、图像描述与文本生成等多种任务。

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

## 环境与依赖

不同实验的依赖略有差异，每个子目录下通常提供独立的 `requirements.txt`。建议为每个实验单独创建 Conda 或虚拟环境。例如：

```bash
cd eighthExperience
conda create -n eighthExperience python=3.10
conda activate eighthExperience
pip install -r requirements.txt
```

其它实验可参考对应子目录下的 `README.md` 或 `requirements.txt`。

## 使用说明

1. 选择感兴趣的实验子目录，例如 `fifthExperience/` 或 `eighthExperience/`；
2. 按该子目录下的 `README.md` 或 `实验报告.md` 中的说明准备数据和环境；
3. 运行训练脚本或推理脚本，例如：

   ```bash
   python train.py
   python eval.py
   python inference.py
   ```

## 版权与致谢

本仓库仅用于课程实验与个人学习研究之用，部分代码参考或改写自公开开源项目（如 YOLOv5、常见图像分割与图像描述实现等）。如需在其它项目中使用，请根据各子项目中的 LICENSE 或原始开源仓库的协议进行引用与遵守。