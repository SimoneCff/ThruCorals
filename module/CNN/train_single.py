import os
import torch
import torchvision.transforms as transforms
from .module.DTS import SeaDataset
from .module.CNN import SeaCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from rich.progress import Progress
import matplotlib.pyplot as plt
import datetime
from multiprocessing import  cpu_count

batch_size = 256
num_epochs = 2
learning_rate = 0.001
rsize = (256,256)
classes = 3

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
    coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_count())

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

    #Creazione Dizionario delle loss
    loss_history = []
    epoch_history = []


    #Train loop
    with Progress() as progress:
        for epoch in range(num_epochs):
            running_loss = 0.0
            task = progress.add_task("Training Model_A", total=len(coral_loader))
            model.train()

            for batch_idx, (images, labels) in enumerate(coral_loader):
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
                
                # Update progress bar
                progress.update(task, advance=1, description=f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx}/{len(coral_loader)}", status=f"Loss: {loss.item():.4f}")

            
            #Loss media
            epoch_loss = running_loss/len(coral_dataset)

            #Salvataggio Loss
            loss_history.append(epoch_loss)
            epoch_history.append(epoch+1)


            print(f"Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.4f}")

            #Save
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'epoch' : epoch
            }, 'checkpoint.pth')


    date = datetime.datetime.now()
    date_str = date.strftime("%d-%m-%Y-%H-%M")

    # plot delle loss
    plt.plot(epoch_history,loss_history)
    plt.title('Train Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_graphic/loss_plot_'+date_str+'.png')
    plt.show()
    
    
    print("Train Done, Exit")


if __name__ == '__main__':
    Start_Train()

