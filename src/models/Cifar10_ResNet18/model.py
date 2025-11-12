import torch.nn as nn
from torchvision import models


def get_cifar10_resnet18() -> models.ResNet:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=10)

    return model

def main():
    model = get_cifar10_resnet18()
    print(model)

if __name__ == "__main__":
    main()

