import torch.nn as nn
from torchvision import models


def get_cifar10_resnet18() -> models.ResNet:
    model = models.resnet18(pretrained=True)
    # 他のパラメータは訓練せず最後の層のみを訓練するため
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=10)

    return model

def main():
    model = get_cifar10_resnet18()
    print(model)

if __name__ == "__main__":
    main()

