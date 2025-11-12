import torch
import torch.nn as nn
from torch.nn.modules import MaxPool2d
from torchtyping import TensorType

class Cifar10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        ) 
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 2 * 2, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x: TensorType["batch", 3, 32, 32]) -> TensorType["batch", 10]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

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

