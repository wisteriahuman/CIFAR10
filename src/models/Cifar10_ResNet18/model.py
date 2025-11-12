import torch.nn as nn
from torchvision import models


def get_cifar10_resnet18() -> models.ResNet:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10)

    return model

def main():
    model = get_cifar10_resnet18()
    print(model)

if __name__ == "__main__":
    main()

