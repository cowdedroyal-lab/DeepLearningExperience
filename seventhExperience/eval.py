# 评估脚本：加载已保存模型，在PTB测试集上计算困惑度
import argparse
import math
import torch

from src import data
from src.model import RNNModel, TransformerModel, PositionalEncoding


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data/ptb', help='PTB data dir')
    ap.add_argument('--checkpoint', type=str, default='model.pt', help='saved model path')
    ap.add_argument('--accel', action='store_true', help='use accelerator')
    ap.add_argument('--model', type=str, default='LSTM')
    ap.add_argument('--bptt', type=int, default=35)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.accel and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    corpus = data.Corpus(args.data)

    def batchify(data_tensor, bsz):
        nbatch = data_tensor.size(0) // bsz
        data_tensor = data_tensor.narrow(0, 0, nbatch * bsz)
        data_tensor = data_tensor.view(bsz, -1).t().contiguous()
        return data_tensor.to(device)

    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size)

    criterion = torch.nn.NLLLoss()

    # 评估函数拷贝自官方示例（略）
    def repackage_hidden(h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data_b = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data_b, target

    def evaluate(data_source, model):
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

    with open(args.checkpoint, 'rb') as f:
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
            with _safe_globals_ctx(safe_globals):  # 与 train.py/inference.py 保持一致
                model = torch.load(f, map_location=device)
        else:
            model = torch.load(f, map_location=device)
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

    test_loss = evaluate(test_data, model)
    print('| test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))


if __name__ == '__main__':
    main()
