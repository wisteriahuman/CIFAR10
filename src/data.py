from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size: int=64) -> tuple[DataLoader[datasets.CIFAR10], DataLoader[datasets.CIFAR10]]:
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(XX: int,), std=(XX: int,))
    ]) 
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(XX: int,), std=(XX: int,))
    ]) 
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    print("=" * 3 + " CIFAR10 " + "=" * 3)
    print(f"train data type: {type(train_data)}")
    print(f"train data size: {train_data[0][0].size()}")

if __name__ == "__main__":
    main()
