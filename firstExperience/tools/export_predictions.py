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


def plot_predictions_grid(model_path: str, out_path: str, n: int = 36, rows: int = 6, cols: int = 6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))

    model = CNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i in range(rows * cols):
        if i >= images.size(0):
            break
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images[i, 0].numpy(), cmap='gray')
        title = f"P:{preds[i].item()} / T:{labels[i].item()}"
        color = 'green' if preds[i].item() == labels[i].item() else 'red'
        ax.set_title(title, color=color, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/best_mnist_cnn.pth')
    parser.add_argument('--out', type=str, default='./figures/predictions_grid.png')
    args = parser.parse_args()

    plot_predictions_grid(args.model, args.out)


if __name__ == '__main__':
    main()
