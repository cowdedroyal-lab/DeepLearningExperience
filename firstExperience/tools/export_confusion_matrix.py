import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def compute_confusion_matrix(model_path: str = './models/best_mnist_cnn.pth', out_path: str = './figures/confusion_matrix.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = CNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_classes = 10
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[t.long(), p.long()] += 1

    cm = conf.cpu().numpy()

    # 归一化到每行（真实标签）
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm[np.isnan(cm_norm)] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6,5))
    im = plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(num_classes), list(range(num_classes)))
    plt.yticks(range(num_classes), list(range(num_classes)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MNIST Confusion Matrix (Row-Normalized)')

    # 在格子里标注数值
    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            color = 'white' if val > thresh else 'black'
            plt.text(j, i, f"{val*100:.0f}%", ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/best_mnist_cnn.pth')
    parser.add_argument('--out', type=str, default='./figures/confusion_matrix.png')
    args = parser.parse_args()

    compute_confusion_matrix(args.model, args.out)
