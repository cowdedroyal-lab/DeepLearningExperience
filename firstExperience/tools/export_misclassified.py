import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.out = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


def export_misclassified(model_path: str, out_path: str, limit: int = 36, rows: int = 6, cols: int = 6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = CNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    images, labels, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            p = logits.argmax(dim=1).cpu()
            mism = (p != y)
            if mism.any():
                idxs = torch.nonzero(mism).view(-1)
                for i in idxs:
                    images.append(x[i, 0].numpy())
                    labels.append(int(y[i].item()))
                    preds.append(int(p[i].item()))
                    if len(images) >= limit:
                        break
            if len(images) >= limit:
                break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i in range(min(limit, len(images))):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"P:{preds[i]} / T:{labels[i]}", color='red', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/best_mnist_cnn.pth')
    parser.add_argument('--out', type=str, default='./figures/misclassified_grid.png')
    parser.add_argument('--limit', type=int, default=36)
    args = parser.parse_args()
    export_misclassified(args.model, args.out, limit=args.limit)
