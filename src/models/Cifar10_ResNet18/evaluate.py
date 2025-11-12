import torch
import torch.nn as nn
from src.models.Cifar10_ResNet18 import get_cifar10_resnet18
from src.data import get_data_loaders


MODEL_PATH = "models/Cifar10_ResNet18.pth"
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_cifar10_resnet18()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    _, test_loader = get_data_loaders()
    loss_total = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for input, label in test_loader:
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            loss_total += loss
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    avg_loss = loss_total / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    evaluate()

if __name__ == "__main__":
    main()
