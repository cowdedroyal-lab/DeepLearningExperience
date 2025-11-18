# ViT模型在CIFAR-10数据集上的训练与可视化

## 简介

这份代码将实现以下功能：
1.  **参数化管理**：使用 `argparse` 模块统一管理所有超参数和配置。
2.  **TensorBoard可视化**：集成 `torch.utils.tensorboard.SummaryWriter` 来记录和可视化训练过程中的损失以及测试集上的准确率变化。
3.  **最佳模型保存**：在训练过程中，实时监控测试集上的准确率，并只保存在准确率达到历史最高时的模型权重。
4.  **代码完整性**：包含PDF文档中提供的所有核心模块，如 `Attention`, `FeedForward`, `Transformer`, 和 `ViT` 的完整实现。
5.  **详细注释**：为关键部分的代码提供清晰的注释，帮助理解。

## 环境配置

1. 创建 conda 环境

```bash
conda create -n secondExperience python=3.10 -y
conda activate secondExperience
```

2. 安装pytorch和torchvision

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

3. 前置依赖库安装

```bash
pip install tensorboard numpy einops
```

---

## 如何运行

1.  打开终端或命令行，进入该文件所在的目录。
2.  执行以下命令开始训练：

    ```bash
    python train_vit_cifar10.py --epochs 100 --batch-size 128 --lr 0.001
    ```

3.  你可以通过命令行参数来调整配置，例如：

    ```bash
    # 使用更深的模型进行训练
    python train_vit_cifar10.py --depth 8 --mlp-dim 2048 --epochs 150
    ```

## **如何查看TensorBoard**

1.  在训练过程中，代码会在当前目录下创建一个 `logs` 文件夹，里面存放着TensorBoard的日志文件。
2.  在终端中打开一个新的标签页或窗口，进入代码所在目录。
3.  运行以下命令启动TensorBoard服务：

    ```bash
    tensorboard --logdir=./logs
    ```

4.  在浏览器中打开TensorBoard提供的URL（通常是 `http://localhost:6006/`），即可看到训练过程中的损失和准确率曲线。

## **结果文件**

*   **模型权重**：训练过程中准确率最高的模型权重会保存在 `checkpoint` 文件夹下的 `vit-best.pth` 文件中。
*   **日志文件**：TensorBoard的日志会保存在 `logs` 文件夹下。