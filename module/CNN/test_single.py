
import os
import torch
import torchvision.transforms as transforms
from module.DTS import SeaDataset
from module.CNN import SeaCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def Start_test():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    nclass = 3
    batch_size = 256
    num_workers = 16

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Dataset
    test_dataset = SeaDataset(data_dir='data/test', transform=transform)

    # Data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model
    model = SeaCNN(num_classes=nclass).to(device)

    # carica i pesi dal checkpoint
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the model

    model.eval()  # Set the model to evaluation mode1

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images : {} %'.format(len(test_dataset), 100 * correct / total))


if __name__ == '__main__':
    Start_test()