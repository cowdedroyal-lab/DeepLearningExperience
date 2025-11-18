import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------- ViT minimal definition (copied to avoid importer side-effects) ----------------

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

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
        return self.mlp_head(x)

# ---------------- Utilities ----------------

def unnormalize(img_tensor, mean, std):
    # img: CxHxW normalized, return HxWxC in [0,1]
    mean = torch.tensor(mean).view(-1,1,1).to(img_tensor.device)
    std = torch.tensor(std).view(-1,1,1).to(img_tensor.device)
    x = img_tensor * std + mean
    x = x.clamp(0,1).detach().cpu().permute(1,2,0).numpy()
    return x


def compute_confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, classes, out_path):
    # normalize by rows (true labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum!=0)
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix (row-normalized)')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor')

    # annotate with counts
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%"
            ax.text(j, i, txt, ha='center', va='center', color='white' if cm_norm[i,j] > thresh else 'black', fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_misclassified_grid(images, labels, preds, classes, out_path, rows=6, cols=6):
    n = min(len(images), rows*cols)
    if n == 0:
        return None
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0))
    axes = axes.flatten()
    for idx in range(rows*cols):
        ax = axes[idx]
        if idx < n:
            ax.imshow(images[idx])
            gt = classes[labels[idx]]
            pd = classes[preds[idx]]
            title = f"P:{pd} | T:{gt}"
            color = 'red' if preds[idx] != labels[idx] else 'green'
            ax.set_title(title, color=color, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Export confusion matrix and misclassified grid for CIFAR-10 ViT model')
    parser.add_argument('--checkpoint', default='checkpoint/vit_cifar10_optimized-best.pth')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # Data transforms matching training test-time
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_root = './data'
    cifar_folder = os.path.join(test_root, 'cifar-10-batches-py')
    need_download = not os.path.isdir(cifar_folder)
    testset = torchvision.datasets.CIFAR10(root=test_root, train=False, download=need_download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    classes = testset.classes

    # Build model
    net = ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=7, heads=8, mlp_dim=1024, dropout=0.2, emb_dropout=0.2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    net.load_state_dict(state)
    net = net.to(args.device)
    net.eval()

    y_true = []
    y_pred = []
    mis_images = []
    mis_labels = []
    mis_preds = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(args.device)
            outputs = net(inputs)
            preds = outputs.argmax(dim=1).cpu()
            targets = targets
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())
            # collect misclassified (after unnormalize)
            wrong = preds.ne(targets).nonzero(as_tuple=False).squeeze(-1).cpu()
            if wrong.numel() > 0:
                for idx in wrong.tolist():
                    img_np = unnormalize(inputs[idx].cpu(), cifar10_mean, cifar10_std)
                    mis_images.append(img_np)
                    mis_labels.append(int(targets[idx].item()))
                    mis_preds.append(int(preds[idx].item()))

    # Confusion matrix
    cm = compute_confusion(np.array(y_true), np.array(y_pred), num_classes=10)
    cm_path = os.path.join(args.figures_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, classes, cm_path)

    # Misclassified grid
    grid_path = os.path.join(args.figures_dir, 'misclassified_grid.png')
    plot_misclassified_grid(mis_images[:36], mis_labels[:36], mis_preds[:36], classes, grid_path, rows=6, cols=6)

    print({'confusion_matrix': cm_path, 'misclassified_grid': grid_path})


if __name__ == '__main__':
    main()
