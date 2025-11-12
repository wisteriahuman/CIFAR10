import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def calculate_mead_std():
    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    imgs = torch.stack([img for img, _ in train_data], dim=0)
    mean = imgs.mean(dim=(0, 2, 3))
    std = imgs.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
def get_data_loaders(batch_size: int=64) -> tuple[DataLoader[datasets.CIFAR10], DataLoader[datasets.CIFAR10]]:
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]) 
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]) 
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    print("=" * 3 + " CIFAR10 " + "=" * 3)
    # print(f"train data type: {type(train_data)}")
    # print(f"train data size: {train_data[0][0].size()}")

    mean, std = calculate_mead_std()
    print(f"Mean: {mean}")
    print(f"Std: {std}")
if __name__ == "__main__":
    main()

