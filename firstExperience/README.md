# 手写数字识别实验代码说明

这份代码实现了手写数字识别的整个流程，并包含了以下功能：
1.  **参数统一管理 (`argparse`)**: 方便地通过命令行调整超参数。
2.  **自动下载数据集**: 如果本地没有MNIST数据集，代码会自动下载。
3.  **模型定义**: 包含了报告中设计的CNN网络结构。
4.  **TensorBoard记录**: 详细记录训练损失和测试准确率，方便可视化分析。
5.  **模型权重保存**: 在训练结束后自动保存效果最好的模型权重。
6.  **设备自动选择**: 自动检测并使用GPU（如果可用），否则使用CPU。
7.  **详细的代码注释**: 对每个部分的功能和关键代码行都进行了详细说明。

---

## **环境准备**

1. 创建conda环境

```bash
conda create -n firstExperience python=3.10 -y
conda activate firstExperience
```

2. 安装必要的Python包

```bash
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0.0"
```

## **如何运行代码**

1.  **打开终端**: 在 `train_mnist.py` 文件所在的目录下打开命令行或终端。

2.  **执行训练**:
    *   **使用默认参数运行**:
        ```bash
        python train_mnist.py
        ```
    *   **自定义参数运行** (例如，训练20轮，批次大小为128):
        ```bash
        python train_mnist.py --epochs 20 --batch-size 128
        ```
    *   **强制使用CPU运行**:
        ```bash
        python train_mnist.py --no-cuda
        ```

## **如何查看TensorBoard**

1.  训练开始后，代码会创建一个 `logs` 文件夹。
2.  在**同级目录**下打开一个新的终端。
3.  输入以下命令并回车：
    ```bash
    tensorboard --logdir=logs
    ```
4.  终端会显示一个本地网址 (通常是 `http://localhost:6006/`)。
5.  将这个网址复制到浏览器中打开，即可看到训练损失和测试准确率随训练轮数变化的图表，以及模型的网络结构图。