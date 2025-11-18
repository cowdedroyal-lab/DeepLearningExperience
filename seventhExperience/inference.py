# 文本采样脚本：从语言模型生成文本（拷贝/改自官方 generate.py）
import argparse
import re
import torch

import src.data as data
from src.model import PositionalEncoding, RNNModel, TransformerModel

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model Text Generation')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=200,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--accel', action='store_true', default=False,
                    help='use accelerator')
parser.add_argument('--model', type=str, default='LSTM')
# 采样与输出控制
parser.add_argument('--topk', type=int, default=0, help='top-k sampling (0 to disable)')
parser.add_argument('--topp', type=float, default=0.0, help='top-p / nucleus sampling (0 to disable)')
parser.add_argument('--keep-unk', action='store_true', default=False, help='keep <unk> tokens in output')
parser.add_argument('--raw', action='store_true', default=False, help='do not pretty-print; keep raw tokens including <eos>')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if args.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3.")

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
        with _safe_globals_ctx(safe_globals):  # copy: guarded deserialization like upstream
            model = torch.load(f, map_location=device)
    else:
        model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
unk_idx = corpus.dictionary.word2idx.get('<unk>')
eos_idx = corpus.dictionary.word2idx.get('<eos>')

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)


def apply_sampling_filters(weights: torch.Tensor) -> torch.Tensor:
    """Apply temperature-scaled unnormalized weights with optional top-k/top-p and unk filtering.

    weights: 1D tensor of non-negative weights (not necessarily normalized)
    returns: filtered weights (still unnormalized), safe for multinomial
    """
    # Skip <unk> if requested: set its weight to 0
    if not args.keep_unk and unk_idx is not None and 0 <= unk_idx < weights.numel():
        weights[unk_idx] = 0.0

    # Top-k filtering
    k = int(args.topk) if args.topk is not None else 0
    if k and k > 0 and k < weights.numel():
        topk_vals, topk_idx = torch.topk(weights, k)
        mask = torch.zeros_like(weights)
        mask.scatter_(0, topk_idx, 1.0)
        weights = weights * mask

    # Top-p (nucleus) filtering
    p = float(args.topp) if args.topp is not None else 0.0
    if p and p > 0.0 and p < 1.0:
        sorted_vals, sorted_idx = torch.sort(weights, descending=True)
        cumulative = torch.cumsum(sorted_vals, dim=0)
        # keep smallest set with cumulative >= p
        keep_mask_sorted = cumulative <= (sorted_vals.sum() * p)
        # Always keep at least the highest-prob token
        if keep_mask_sorted.numel() > 0:
            keep_mask_sorted[0] = True
        keep_mask = torch.zeros_like(weights, dtype=torch.bool)
        keep_mask.scatter_(0, sorted_idx, keep_mask_sorted)
        weights = torch.where(keep_mask, weights, torch.zeros_like(weights))

    # Avoid all-zero
    if torch.all(weights <= 0):
        # Fallback to uniform over vocab except <unk>
        weights = torch.ones_like(weights)
        if not args.keep_unk and unk_idx is not None and 0 <= unk_idx < weights.numel():
            weights[unk_idx] = 0.0
    return weights


def detokenize_ptb(tokens):
    """Simple PTB-style detokenizer: 
    - Render <eos> as newline
    - Drop <unk> (unless --keep-unk)
    - Fix common spacing around punctuation and contractions
    """
    # Join with spaces first
    text = ' '.join(tokens)
    # <eos> to newline
    text = text.replace(' <eos> ', '\n').replace(' <eos>', '\n').replace('<eos> ', '\n').replace('<eos>', '\n')
    # common contractions
    text = (text
            .replace(" n't", "n't")
            .replace(" 's", "'s")
            .replace(" 're", "'re")
            .replace(" 've", "'ve")
            .replace(" 'll", "'ll")
            .replace(" 'm", "'m")
            .replace(" 'd", "'d"))
    # remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\s+([)\]\}])", r"\1", text)
    # remove spaces after opening brackets
    text = re.sub(r"([([\{])\s+", r"\1", text)
    # collapse multiple spaces and trim lines
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    # Ensure each line ends with punctuation
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and line[-1] not in '.!?':
            line += '.'
        lines.append(line)
    return '\n'.join(lines).strip() + '\n'

tokens = []
with torch.no_grad():  # no tracking history
    for i in range(args.words):
        if is_transformer_model:
            output = model(input, False)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_weights = apply_sampling_filters(word_weights)
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)
        else:
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_weights = apply_sampling_filters(word_weights)
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

        # 记录 token
        tok = corpus.dictionary.idx2word[word_idx]
        if not args.keep_unk and tok == '<unk>':
            # 跳过未知词
            pass
        else:
            tokens.append(tok)

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))

text = ' '.join(tokens) if args.raw else detokenize_ptb(tokens)
with open(args.outf, 'w') as outf:
    outf.write(text)
print(f'| Saved generated text to {args.outf}')
