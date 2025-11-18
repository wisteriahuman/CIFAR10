import torch
import torch.nn as nn
from torchtyping import TensorType


class Cifar10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ResidualBlock(in_channels=3, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(in_channels=256, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=512),
            ResidualBlock(in_channels=512, out_channels=512),
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 4 * 4, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x: TensorType["batch", 3, 32, 32]) -> TensorType["batch", 10]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv_block(x)
        out += identity
        out = self.relu(out)
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Cifar10_CNN()
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"出力shape: {output.shape}")
    print(f"出力: {output}")

if __name__ == "__main__":
    main()

