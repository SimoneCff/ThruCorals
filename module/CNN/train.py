import os
import torch
import torchvision.transforms as transforms
from module.DTS import SeaDataset
from module.CNN import SeaCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

batch_size = 32
num_epochs = 10
learning_rate = 0.001
rsize = (256,256)
classes = 10

def Start_Train():
    #Transform Function
    transform = transforms.Compose([
        transforms.Resize(rsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Dataset
    coral_dataset = SeaDataset(data_dir='data/train', transform=transform)
    coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #CNN
    model = SeaCNN(num_classes=classes)

    #Fun Loss & Opt
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

    #Check esistenza
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    #Train loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(coral_loader, desc=f"Epoch (epoch + 1)/{num_epochs}"):
            #Forward pass
            outputs = model(images)

            #loss calc
            loss = criterion(outputs, labels)

            #Gradient Reset
            optimizer.zero_grad()

            #Backpass & Weights
            loss.backward()
            optimizer.step()

            #Running loss
            running_loss += loss.item() * images.size(0)
        
        #Loss media
        epoch_loss = running_loss/len(coral_dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.4f}")

        #Save
        torch.save({
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'epoch' : epoch
        }, 'checkpoint.pth')

    print("Train Done, Exit")


if __name__ == '__main__':
    Start_Train()

