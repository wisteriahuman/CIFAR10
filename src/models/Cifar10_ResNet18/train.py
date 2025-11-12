import torch
import torch.nn as nn
import torch.optim as optim
from src.models.Cifar10_ResNet18 import get_cifar10_resnet18
from src.data import get_data_loaders


MAX_EPOCH = 30

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_cifar10_resnet18()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    train_loader, _ = get_data_loaders()
    num_batches_per_epoch = len(train_loader)
    model.train()
    for epoch in range(MAX_EPOCH):
        loss_total = 0
        for input, label in train_loader:
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{MAX_EPOCH}], Loss: {loss_total/num_batches_per_epoch:.3f}")

    torch.save(model.state_dict(), 'src/model/cifar10_resnet18.pth')


def main():
    import time

    print("訓練を開始します")
    start = time.time()
    train()
    end = time.time()
    print("訓練が完了しました")
    print(f"訓練時間: {end - start:.2f} sec")

if __name__ == "__main__":
    main()
