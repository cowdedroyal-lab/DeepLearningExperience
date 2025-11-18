# -----------------------------------------------------------------------------
# 实验三：自动写诗 - 完整 PyTorch 实现 (最终优化版)
#
# 包含：
# 1. [新增] 使用 argparse 统一管理超参数
# 2. TensorBoard 集成，用于记录训练 loss 和验证集 Perplexity
# 3. 完整的模型训练、验证和预测流程
# 4. 详细的注释
# 5. Top-k 采样和温度系数，解决生成重复问题
# 6. 学习率调度器、权重衰减和梯度裁剪，缓解过拟合
# 7. 训练过程中自动保存最佳模型的功能
# 8. [新增] 提前停止 (Early Stopping) 策略，有效防止过拟合
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import time
import torch.nn.functional as F
import argparse # 引入 argparse 模块

# --- 1. 使用 argparse 统一管理所有超参数和配置 ---
def get_args():
    """
    定义并解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="PyTorch Poetry Generation Model")

    # 数据集参数
    parser.add_argument('--dataset-path', type=str, default='tang.npz', help='数据集文件路径')
    parser.add_argument('--validation-split', type=float, default=0.1, help='验证集所占比例')

    # 模型参数
    parser.add_argument('--embedding-dim', type=int, default=256, help='词嵌入维度')
    parser.add_argument('--hidden-dim', type=int, default=512, help='LSTM 隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=2, help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout 概率')
    # 双向 LSTM 在自回归生成任务中通常会泄露未来信息，默认关闭。
    parser.add_argument('--bidirectional', action='store_true', default=False, help='是否使用双向 LSTM（默认: False）')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='优化器选择')
    parser.add_argument('--weight-tying', action='store_true', default=False, help='是否在输出层与 embedding 之间共享权重')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (设置较大值，配合提前停止)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler-patience', type=int, default=3, help='学习率调度器耐心值')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='学习率调度器衰减因子')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='提前停止的耐心值') # 新增

    # 预测参数
    parser.add_argument('--start-words', type=str, default="湖光秋月两相和", help='生成诗歌的起始句')
    parser.add_argument('--max-gen-len', type=int, default=64, help='生成诗歌的最大长度')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k 采样参数')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度系数')

    # 环境与路径参数
    parser.add_argument('--no-cuda', action='store_true', help='不使用 CUDA')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='模型保存目录')

    args = parser.parse_args()
    args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    
    return args

# --- 2. 数据准备 ---
def load_data(args):
    """
    加载并准备数据，包括划分训练集和验证集。
    """
    print("开始加载数据集...")
    datas = np.load(args.dataset_path, allow_pickle=True)
    data = datas['data']
    word2ix = datas['word2ix'].item()
    ix2word = datas['ix2word'].item()
    
    data = torch.from_numpy(data).long()
    
    num_total = len(data)
    num_val = int(num_total * args.validation_split)
    num_train = num_total - num_val
    
    train_data, val_data = torch.utils.data.random_split(data, [num_train, num_val])
    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"数据集加载完成。训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    print(f"词汇表大小: {len(word2ix)}")
    
    return train_dataloader, val_dataloader, word2ix, ix2word

# --- 3. 模型构建 ---
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, args):
        super(PoetryModel, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_directions = 2 if args.bidirectional else 1

        self.embeddings = nn.Embedding(vocab_size, args.embedding_dim)
        # 在 embedding 之后增加 dropout，有助于正则化并减少生成时的重复
        self.embedding_dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(args.embedding_dim,
                            args.hidden_dim,
                            num_layers=args.num_layers,
                            batch_first=True,
                            dropout=args.dropout,
                            bidirectional=args.bidirectional)

        # 输出层：支持可选的权重共享 (weight tying)
        self.weight_tying = args.weight_tying
        if self.weight_tying:
            # 为了兼容 embedding_dim 与 hidden_dim * directions 不一致的情况，
            # 使用一个投影层将 LSTM 输出映射到 embedding_dim，再与 embedding.weight 做矩阵相乘产生 logits
            self.proj = nn.Linear(args.hidden_dim * self.num_directions, args.embedding_dim)
            # 偏置项
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.linear = nn.Linear(args.hidden_dim * self.num_directions, vocab_size)

    def forward(self, input_seq, hidden_state=None):
        embeds = self.embeddings(input_seq)
        embeds = self.embedding_dropout(embeds)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)

        batch_size, seq_len, _ = lstm_out.shape
        output = lstm_out.reshape(batch_size * seq_len, -1)

        if self.weight_tying:
            proj_out = self.proj(output)  # (batch*seq, embedding_dim)
            # 与 embedding 权重相乘得到 vocab 大小的 logits
            logits = torch.matmul(proj_out, self.embeddings.weight.t()) + self.output_bias
        else:
            logits = self.linear(output)

        return logits, hidden_state

# --- 4. 模型预测 ---
def generate(model, start_words, ix2word, word2ix, args):
    """
    根据给定的起始句生成一首诗。
    """
    model.eval()
    results = list(start_words)
    start_words_len = len(start_words)
    
    # 如果数据集中没有特殊 token，使用索引 0 作为回退（通常是 <PAD> 或其它）。
    start_idx = word2ix.get('<START>', 0)
    input_tensor = torch.tensor([start_idx], dtype=torch.long, device=args.device).view(1, 1)
    hidden = None

    with torch.no_grad():
        for i in range(args.max_gen_len):
            output, hidden = model(input_tensor, hidden)
            
            if i < start_words_len:
                # 处理 start_words 中可能出现的未登录词
                w = results[i]
                idx = word2ix.get(w, word2ix.get('<UNK>', start_idx))
                input_tensor = torch.tensor([idx], dtype=torch.long, device=args.device).view(1, 1)
            else:
                output = output.squeeze() / max(args.temperature, 1e-8)
                vocab_size = output.size(-1)
                k = min(args.top_k, vocab_size)
                top_k_logits, top_k_indices = torch.topk(output, k)
                top_k_probs = F.softmax(top_k_logits, dim=-1)

                sampled_index_in_top_k = torch.multinomial(top_k_probs, 1).item()
                final_index = top_k_indices[sampled_index_in_top_k].item()

                w = ix2word[final_index]
                if w == '<EOP>':
                    break
                results.append(w)
                input_tensor = torch.tensor([final_index], dtype=torch.long, device=args.device).view(1, 1)

    return "".join(results)

# --- 5. 验证函数 ---
def evaluate(model, dataloader, criterion, args):
    """
    在验证集上评估模型。
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for data_batch in dataloader:
            inputs = data_batch[:, :-1].to(args.device)
            targets = data_batch[:, 1:].to(args.device)

            logits, _ = model(inputs)
            loss = criterion(logits, targets.reshape(-1))
            
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

# --- 6. 训练函数 ---
def train(args):
    """
    主训练流程
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f'runs/poetry_gen_{int(time.time())}')
    
    train_loader, val_loader, word2ix, ix2word = load_data(args)
    vocab_size = len(word2ix)

    model = PoetryModel(vocab_size, args)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    # 选择优化器：Adam 或 AdamW
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                     patience=args.scheduler_patience, 
                                                     factor=args.scheduler_factor, 
                                                     verbose=True)
    
    print("\n开始训练...")
    global_step = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0 # [新增] 用于提前停止的计数器

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_data in progress_bar:
            inputs = batch_data[:, :-1].to(args.device)
            targets = batch_data[:, 1:].to(args.device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets.reshape(-1))
            loss.backward()
            # 记录并裁剪梯度范数以便诊断
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 把梯度范数写入 TensorBoard（每个 step）
            try:
                writer.add_scalar('GradNorm/train', grad_norm, global_step)
            except Exception:
                pass
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
            
            if global_step % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
        
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, args)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Perplexity/validation', val_perplexity, epoch)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} 完成. Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 # [新增] 重置计数器
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"发现更好的模型，验证集损失降至 {best_val_loss:.4f}，已保存至 {save_path}")
        else:
            epochs_no_improve += 1 # [新增] 验证集损失没有改善，计数器加一
        
        # [新增] 提前停止逻辑
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"验证集损失在 {args.early_stopping_patience} 个 epochs 内没有改善，触发提前停止。")
            break

        # 在每个 epoch 结束后都生成一首诗，观察模型变化
        generated_poetry = generate(model, args.start_words, ix2word, word2ix, args)
        print(f"示例生成: '{args.start_words}' -> '{generated_poetry}'\n")
        writer.add_text('Generated Poetry', generated_poetry, epoch)

    writer.close()
    
    final_model_path = "poetry_model_final.pth"
    # 如果训练提前停止，我们仍然保存最后的模型状态，但最佳模型已经在过程中保存了
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成，最终模型已保存至 {final_model_path}")
    print(f"最佳模型保存在: {os.path.join(args.checkpoint_dir, 'best_model.pth')}，其验证集损失为: {best_val_loss:.4f}")

# --- 主程序入口 ---
if __name__ == '__main__':
    args = get_args()
    
    # 打印所有配置
    print("当前配置:")
    for arg in vars(args):
        print(f"  - {arg}: {getattr(args, arg)}")
    
    train(args)