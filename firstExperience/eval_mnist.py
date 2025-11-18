import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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


def evaluate(model_path: str = './models/best_mnist_cnn.pth'):
    device = torch.device('cpu')

    # dataset
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # model
    model = CNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = 100.0 * correct / total
    print(f'Test Accuracy: {acc:.2f}%  ({correct}/{total})')


if __name__ == '__main__':
    evaluate()
