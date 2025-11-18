# 唐诗生成实验说明
## 简介
这份代码严格遵循了您的四点要求：
1.  **使用 `Config` 类统一管理变量**，便于参数的集中修改和维护。
2.  **集成 TensorBoard**，用于可视化训练过程中的损失（loss）和在验证集上的困惑度（Perplexity），困惑度是衡量语言模型好坏的标准指标，比准确率更适合此任务。
3.  **代码结构和逻辑参考了 PDF**，并在此基础上进行了扩充和完善，使其成为一个完整的项目。
4.  **提供了详细的注释**，解释了每个部分的功能和关键代码行的作用。

---

## 环境配置

1. 创建 conda 环境

```bash
conda create -n thirdExperience python=3.10 -y
conda activate thirdExperience
```

2. 安装 pytorch

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

3. 前置依赖库安装

```bash
pip install tensorboard numpy tqdm
```

4. 将下载的唐诗数据集 `tang.npz` 文件与下面的 `main.py` 脚本放在同一个目录下。

## 运行实验代码：`main.py`
1.  运行脚本：

    ```bash
    python main.py
    ```

2.  在终端中会看到一个新的 `runs` 目录被创建，使用 TensorBoard 查看训练过程：

    ```bash
    tensorboard --logdir=runs
    ```
    
    然后在浏览器中打开显示的网址（通常是 `http://localhost:6006/`）。

