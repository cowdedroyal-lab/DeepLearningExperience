# 实验7：神经网络语言模型（PTB + LSTM, PyTorch）

本项目实现 PTB（Penn Treebank）语料上的循环神经网络语言模型（LSTM），训练目标为困惑度 PPL < 80。实现严格参考并复用 PyTorch 官方 examples 的 word_language_model 代码（BSD-3-Clause 许可），复杂核心代码已直接复制并标注 copy 来源，保留原始逻辑与超参数接口。

- 上游参考：PyTorch Examples / word_language_model
  - 仓库：https://github.com/pytorch/examples/tree/main/word_language_model
  - 许可：BSD 3-Clause（本项目根目录已附 LICENSE）
- 复制文件（均在 `src/` 下，文件头标注了 copy 来源）：
  - `src/data.py`（数据读取、词表构建）
  - `src/model.py`（RNN/LSTM/GRU/Transformer 模型）
- 训练脚本 `train.py` 的训练/评估核心循环也参考并复制了上游逻辑（已在相应函数处标注 copy）。

## 环境要求

- Python 3.8+
- PyTorch 1.12+（CPU 或 GPU 均可）
- 其他依赖见 `requirements.txt`

可选：若使用 GPU，请根据你的 CUDA 版本安装匹配的 torch 预编译包。

## 数据准备（PTB）

本项目提供数据下载脚本，会从 Mikolov 网站下载 `simple-examples.tgz`，解压后把 `ptb.{train,valid,test}.txt` 重命名为 `train.txt/valid.txt/test.txt`，以适配 PyTorch examples 的 `Corpus` 读取格式。

```bash
# 在 seventhExperience 目录下执行：
python tools/download_ptb.py
```

下载完成后，数据位于：`./data/ptb/{train.txt, valid.txt, test.txt}`。

## 训练（目标 PPL < 80）

推荐超参数（来自经典 PTB-LSTM large 设置），一般可在数十个 epoch 后达到 <80 PPL：

```bash
python train.py \
  --data ./data/ptb \
  --model LSTM \
  --emsize 1500 --nhid 1500 --nlayers 2 \
  --dropout 0.65 --bptt 35 --batch_size 20 \
  --lr 20 --epochs 40 --tied

```

训练过程中会在验证集上早停保存最佳模型到 `model.pt`，并在训练结束对测试集打印最终困惑度（PPL）。

注：如想更稳定的优化，可加 `--use-optimizer` 使用 AdamW 优化器；但为尽量复现上游实现的结果，默认仍提供 SGD + 退火策略。

## 评估

```bash
python eval.py --data ./data/ptb --checkpoint model.pt --model LSTM
```

输出包含 test loss 与 test ppl。

## 文本生成（采样）

```bash
python inference.py --data ./data/ptb --checkpoint model.pt --model LSTM --words 200
```

生成文本写入 `generated.txt`。推理脚本做了更友好的输出：

- 默认跳过 `<unk>`，将 `<eos>` 渲染为换行，并进行简单 detokenize，输出更自然。
- 可选采样控制：`--temperature`、`--topk`、`--topp`（nucleus）。
- 想要原始 token 序列（包含 `<eos>`/`<unk>`），可加 `--raw`；若保留 `<unk>`，加 `--keep-unk`。

## 项目结构

- `train.py`：训练与验证（核心循环 copy 自上游示例，保留退火与 clip）
- `eval.py`：加载已训好模型做测试集评估
- `inference.py`：基于已训好模型随机采样生成文本
- `src/data.py`：数据与词表（copy）
- `src/model.py`：模型定义（RNN/LSTM/GRU/Transformer，copy）
- `tools/download_ptb.py`：数据下载与整理
- `requirements.txt`：最小依赖
- `LICENSE`：BSD-3-Clause（与上游一致）

## 实验报告要点建议

- 简述语言模型与困惑度定义（PPL = exp(cross-entropy)）
- 说明 RNN/LSTM 原理、BPTT、梯度裁剪、dropout、权重共享（tied）
- 描述数据准备、模型结构与关键超参（layers=2, hidden=1500, bptt=35, dropout=0.65, lr 退火/AdamW）
- 训练曲线与 PPL 随 epoch 变化图（可记录每 epoch 的 valid ppl）
- 在测试集上报告最终 PPL，并与 baseline 或论文结果对比
- 附录超参数、运行截图与生成样例

## 致谢与版权

- 大段核心代码 copy 自 PyTorch 官方 `examples/word_language_model`，原始仓库采用 BSD 3-Clause 许可。本项目保留并附带 LICENSE，并在相关文件头标注了来源与 copy 声明。
