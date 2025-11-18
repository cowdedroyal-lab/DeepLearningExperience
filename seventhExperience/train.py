# 实验7：PTB上基于LSTM的神经语言模型（PyTorch）
# 复杂核心训练/评估代码 copy 自 PyTorch 官方 examples/word_language_model（BSD-3-Clause）并做少量适配
# 来源：https://github.com/pytorch/examples/tree/main/word_language_model

import argparse
import time
import math
import os
import torch
import torch.nn as nn

from src import data
from src.model import RNNModel, TransformerModel, PositionalEncoding


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='./data/ptb',
                        help='location of the data corpus (contains train.txt, valid.txt, test.txt)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=1500,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1500,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.65,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--accel', action='store_true',
                        help='Enables accelerator (CUDA/MPS/XPU if available)')
    parser.add_argument('--use-optimizer', action='store_true',
                        help='Uses AdamW optimizer (default is SGD)')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.accel and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cuda")
    print('Using device:', device)

    # -------------------------------
    # 数据加载 copy 自官方示例 (轻微适配路径)
    # -------------------------------
    corpus = data.Corpus(args.data)

    def batchify(data_tensor, bsz):
        # copy from examples/word_language_model/main.py
        nbatch = data_tensor.size(0) // bsz
        data_tensor = data_tensor.narrow(0, 0, nbatch * bsz)
        data_tensor = data_tensor.view(bsz, -1).t().contiguous()
        return data_tensor.to(device)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    # -------------------------------
    # 构建模型 copy 自官方示例 (保持API)
    # -------------------------------
    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = TransformerModel(ntokens, args.emsize, 2, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    criterion = nn.NLLLoss()

    if args.use_optimizer:
        # 使用 AdamW 时对 lr 做保护：上游默认 lr=20 适用于手写 SGD 更新（示例中的手动 p.data.add_），
        # 但对自适应优化器（Adam/AdamW）这是过大的学习率，会导致 loss 发散为 inf/NaN 并在 math.exp 时溢出。
        # 如果用户传入了非常大的 lr（>1.0），这里将 lr 降至更合适的默认 1e-3 并打印警告。
        if args.lr > 1.0:
            safe_lr = 1e-3
            print(f"Warning: --use-optimizer set but lr={args.lr} is very large for AdamW; using lr={safe_lr} for optimizer instead.")
        else:
            safe_lr = args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=safe_lr)
    else:
        optimizer = None

    def _safe_exp(x: float) -> float:
        """Safely compute exp(x) returning inf on overflow instead of raising."""
        try:
            return math.exp(x)
        except OverflowError:
            return float('inf')

    # -------------------------------
    # 训练/评估实用函数 copy 自官方示例 (保持逻辑)
    # -------------------------------
    def repackage_hidden(h):
        # copy from examples/word_language_model/main.py
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def get_batch(source, i):
        # copy from examples/word_language_model/main.py
        seq_len = min(args.bptt, len(source) - 1 - i)
        data_b = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data_b, target

    def evaluate(data_source):
        # copy from examples/word_language_model/main.py
        model.eval()
        total_loss = 0.
        ntokens_local = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data_b, targets = get_batch(data_source, i)
                if args.model == 'Transformer':
                    output = model(data_b)
                    output = output.view(-1, ntokens_local)
                else:
                    output, hidden = model(data_b, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data_b) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    def train_loop():
        # copy from examples/word_language_model/main.py (略微简化打印)
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens_local = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = model.init_hidden(args.batch_size)
        lr = args.lr
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data_b, targets = get_batch(train_data, i)
            if args.use_optimizer:
                optimizer.zero_grad()
            else:
                model.zero_grad()
            if args.model == 'Transformer':
                output = model(data_b)
                output = output.view(-1, ntokens_local)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data_b, hidden)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if args.use_optimizer:
                optimizer.step()
            else:
                for p in model.parameters():
                    p.data.add_(p.grad, alpha=-lr)
            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                # 若 cur_loss 非有限（inf/NaN），提前终止并提示，避免 math.exp 抛出 OverflowError
                if not math.isfinite(cur_loss):
                    print('\nError: non-finite training loss detected (cur_loss={}). Stopping training.'.format(cur_loss))
                    return
                ppl_val = _safe_exp(cur_loss)
                print('| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    batch, len(train_data) // args.bptt, lr, elapsed * 1000 / args.log_interval, cur_loss, ppl_val))
                total_loss = 0
                start_time = time.time()

    # -------------------------------
    # 训练过程 + 验证早停 + 测试
    # -------------------------------
    best_val_loss = None
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loop()
            val_loss = evaluate(val_data)
            print('-' * 89)
            val_ppl = _safe_exp(val_loss) if math.isfinite(val_loss) else float('inf')
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, val_ppl))
            print('-' * 89)
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # 学习率退火 (copy 自官方逻辑)
                args.lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # 测试集评估
    with open(args.save, 'rb') as f:
        if args.model == 'Transformer':
            safe_globals = [
                PositionalEncoding,
                TransformerModel,
                torch.nn.functional.relu,
                torch.nn.modules.activation.MultiheadAttention,
                torch.nn.modules.container.ModuleList,
                torch.nn.modules.dropout.Dropout,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                torch.nn.modules.normalization.LayerNorm,
                torch.nn.modules.sparse.Embedding,
                torch.nn.modules.transformer.TransformerEncoder,
                torch.nn.modules.transformer.TransformerEncoderLayer,
            ]
        else:
            safe_globals = [
                RNNModel,
                torch.nn.modules.dropout.Dropout,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.rnn.GRU,
                torch.nn.modules.rnn.LSTM,
                torch.nn.modules.rnn.RNN,
                torch.nn.modules.sparse.Embedding,
            ]
        # 兼容旧版 PyTorch：无 torch.serialization.safe_globals 时回退为常规 torch.load
        try:
            from torch.serialization import safe_globals as _safe_globals_ctx  # type: ignore[attr-defined]
            use_safe_ctx = True
        except Exception:
            use_safe_ctx = False

        if use_safe_ctx:
            with _safe_globals_ctx(safe_globals):  # copy: guarded deserialization like upstream
                model_loaded = torch.load(f, map_location=device)
        else:
            model_loaded = torch.load(f, map_location=device)

    # 用最佳检查点进行测试评估
    model = model_loaded.to(device)
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()
    test_loss = evaluate(test_data)
    test_ppl = _safe_exp(test_loss) if math.isfinite(test_loss) else float('inf')
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, test_ppl))
    print('=' * 89)


if __name__ == '__main__':
    main()
