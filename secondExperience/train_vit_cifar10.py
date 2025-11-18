import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment
import os
import argparse
import time

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------
# 1. 参数解析器 (args)
# ----------------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ViT Training')

# 数据集和模型参数
parser.add_argument('--image-size', default=32, type=int, help='Image size for ViT input')
parser.add_argument('--patch-size', default=4, type=int, help='Patch size for image splitting')

# 训练超参数
parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
# 【MODIFIED】更新权重衰减的默认值以匹配AdamW的最佳实践
parser.add_argument('--weight-decay', default=0.05, type=float, help='Weight decay')
parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
parser.add_argument('--batch-size', default=128, type=int, help='Training batch size')

# 【MODIFIED】更新ViT模型结构的默认参数，使其更强大
parser.add_argument('--dim', default=512, type=int, help='Embedding dimension')
parser.add_argument('--depth', default=7, type=int, help='Depth of Transformer (number of layers)')
parser.add_argument('--heads', default=8, type=int, help='Number of attention heads')
parser.add_argument('--mlp-dim', default=1024, type=int, help='Dimension of the MLP layer in Transformer')

# 运行和保存设置
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--log-dir', default='logs', type=str, help='Directory for TensorBoard logs')
parser.add_argument('--checkpoint-dir', default='checkpoint', type=str, help='Directory for saving checkpoints')
parser.add_argument('--net', default='vit_cifar10_optimized', type=str, help='Network name for saving checkpoints')

args, unknown = parser.parse_known_args()


# ----------------------------------
# 2. 辅助函数和工具 (保持不变)
# ----------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def progress_bar(batch_idx, total_batches, msg):
    term_width = 80
    done = int(term_width * (batch_idx + 1) / total_batches)
    percent = (batch_idx + 1) / total_batches * 100
    bar = f"[{'=' * done}{' ' * (term_width - done - 2)}] {percent:.2f}% - {msg}"
    print(f"\r{bar}", end="")
    if batch_idx + 1 == total_batches:
        print()


# ----------------------------------
# 3. ViT 模型定义 (保持不变)
# ----------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# ----------------------------------
# 4. 训练与验证函数 (保持不变)
# ----------------------------------
def train(epoch, net, trainloader, optimizer, criterion, device, writer):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        msg = f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})'
        progress_bar(batch_idx, len(trainloader), msg)

    avg_loss = train_loss / len(trainloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', 100.*correct/total, epoch)


def test(epoch, net, testloader, criterion, device, writer, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            msg = f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})'
            progress_bar(batch_idx, len(testloader), msg)
            
    avg_loss = test_loss / len(testloader)
    acc = 100. * correct / total
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', acc, epoch)

    if acc > best_acc:
        print('Saving best model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        torch.save(state, os.path.join(args.checkpoint_dir, f'{args.net}-best.pth'))
        best_acc = acc
    
    return best_acc


# ----------------------------------
# 5. 主执行函数
# ----------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    log_path = os.path.join(args.log_dir, f"{args.net}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_path)
    
    # --- 数据加载与预处理 ---
    print('==> Preparing data..')
    
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # 【MODIFIED】启用AutoAugment以实现更强的数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # 启用AutoAugment
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        # 可选：如果过拟合仍然严重，可以启用RandomErasing (Cutout)
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # --- 模型初始化 ---
    print('==> Building model..')
    net = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        # 【MODIFIED】增加Dropout率以匹配更深的模型，防止过拟合
        dropout=0.2,
        emb_dropout=0.2
    )
    net = net.to(device)

    # --- 从checkpoint恢复训练 ---
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.net}-best.pth')
        assert os.path.exists(checkpoint_path), f'Error: no checkpoint found at {checkpoint_path}'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        
    # --- 损失函数、优化器和学习率调度器 ---
    `criterion` = nn.CrossEntropyLoss()
    # 【MODIFIED】改用 AdamW 优化器，它通常在Transformer上表现更好
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # --- 训练和验证循环 ---
    for epoch in range(start_epoch, args.epochs):
        train(epoch, net, trainloader, optimizer, criterion, device, writer)
        best_acc = test(epoch, net, testloader, criterion, device, writer, best_acc)
        scheduler.step()
        print(f"Current best accuracy: {best_acc:.3f}%")
        
    writer.close()
    print(f"Training finished. Best accuracy: {best_acc:.3f}%")


if __name__ == '__main__':
    main()