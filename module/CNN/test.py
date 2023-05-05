
import os
import torch
import torchvision.transforms as transforms
from .module.DTS import SeaDataset
from .module.CNN import SeaCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def Start_test(path_in, path_in_transf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    nclass = 10
    batch_size = 32

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Dataset
    test_notransform_dataset = SeaDataset(data_dir=path_in, transform=transform)
    test_transform_dataset = SeaDataset(data_dir=path_in_transf, transform=transform)

    # Data loader
    test_notransform_loader = DataLoader(dataset=test_notransform_dataset, batch_size=batch_size, shuffle=True)
    test_transform_loader = DataLoader(dataset=test_transform_dataset, batch_size=batch_size, shuffle=True)

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
        for images, labels in test_notransform_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images not transformed: {} %'.format(len(test_notransform_dataset), 100 * correct / total))

        correct = 0
        total = 0
        for images, labels in test_transform_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images transformed: {} %'.format(len(test_transform_dataset), 100 * correct / total))

if __name__ == '__main__':
    Start_test()