from logging import critical
import torch
import torch.nn as nn
from models.cifar10_cnn import Cifar10_CNN
from data import get_data_loaders
from train import BATCH_SIZE


MODEL_PATH = "models/cifar10_cnn.pth"

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")
    model = Cifar10_CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    _, test_loader = get_data_loaders()
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for input, label in test_loader:
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output, label)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

def main():
    evaluate()

if __name__ == "__main__":
    main()
