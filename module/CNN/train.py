import os
import torch
import torchvision.transforms as transforms
from .module.DTS import SeaDataset
from .module.CNN import SeaCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TimeElapsedColumn
import matplotlib.pyplot as plt
import datetime
from multiprocessing import  cpu_count
import threading
import time

batch_size = 256
num_epochs = 10
learning_rate = 0.001
rsize = (256,256)
classes = 3
work = cpu_count()



def loop(dataset, dataloader, epochs, criterion, optimizer, model, history, model_type, checkpoint_name, progress, task):
    with progress:
        start_time_tot = time.time()
        for epoch in range(epochs):
            progress.reset(task)
            # First Train
            running_loss = 0.0
            model.train()

            start_time_ep = time.time()
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Forward pass
                outputs = model(images)

                # Loss calculation
                loss = criterion(outputs, labels)

                # Gradient Reset
                optimizer.zero_grad()

                # Backpass & Weights
                loss.backward()
                optimizer.step()

                # Running loss
                running_loss += loss.item() * images.size(0)

                # Update progress bar
                progress.update(task, advance=1)

            epoch_loss = running_loss / len(dataset)
            history.append(epoch_loss)
            progress.update(task, completed=len(dataloader))

            end_time_ep = time.time()
            print(f"Epoch {epoch+1}/{epochs} model: {model_type} loss: {epoch_loss:.4f}, Total time Epoch: {end_time_ep - start_time_ep:.2f} s")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_name)
    end_time_tot = time.time()
    print(f"Total Time of training model: {model_type} of total epoch: {epochs} is:= {end_time_tot - start_time_tot:.2f} s")
 

def Start_Train(path_in, path_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Transform Function
    transform = transforms.Compose([
        transforms.Resize(rsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    #Dataset
    coralno_dataset = SeaDataset(data_dir=path_in, transform=transform)
    coralno_loader = DataLoader(dataset=coralno_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=work)

    coralsi_dataset = SeaDataset(data_dir=path_out, transform=transform)
    coralsi_loader = DataLoader(dataset=coralsi_dataset, shuffle=True,batch_size=batch_size, num_workers=work, drop_last=True)

    #CNN
    model_a = SeaCNN(num_classes=classes).to(device)
    model_b = SeaCNN(num_classes=classes).to(device)

    #Fun Loss & Opt
    criterion = nn.CrossEntropyLoss()
    optimizer_a = optim.Adam(model_a.parameters(), lr=learning_rate)
    optimizer_b = optim.Adam(model_b.parameters(), lr=learning_rate)

    #Check esistenza
    if os.path.exists('checkpoint_a.pth'):
        checkpoint = torch.load('checkpoint_a.pth')
        model_a.load_state_dict(checkpoint['model_state_dict'])
        optimizer_a.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if os.path.exists('checkpoint_b.pth'):
        checkpoint = torch.load('checkpoint_b.pth')
        model_b.load_state_dict(checkpoint['model_state_dict'])
        optimizer_b.load_state_dict(checkpoint['optimizer_state_dict'])
    

    #Creazione Dizionario delle loss
    loss_history_a = []
    loss_history_b = []
    epoch_history = list(range(1, num_epochs+1))

    #Creazione Progress Bar
    progress = Progress("[progress.description]{task.description}", BarColumn(), TextColumn("[bold green]{task.completed}/{task.total}"),"|",TimeElapsedColumn() ,"||",TimeRemainingColumn(),"|")

    #Creazione Task
    task_a = progress.add_task("[cyan]Training Model A...", total=len(coralno_loader), task_id="a")
    task_b = progress.add_task("[magenta]Training Model B...", total=len(coralsi_loader), task_id="b")

    #Creazione Threads
    thread_a = threading.Thread(target=loop, args=(coralno_dataset, coralno_loader, num_epochs, criterion, optimizer_a, model_a, loss_history_a, "A", 'checkpoint_a.pth', progress, task_a))
    thread_b = threading.Thread(target=loop, args=(coralsi_dataset, coralsi_loader, num_epochs, criterion, optimizer_b, model_b, loss_history_b, "B", 'checkpoint_b.pth', progress, task_b))
    
    #Start Threads
    thread_a.start()
    thread_b.start()

    #Riunificazione Threads
    thread_a.join()
    thread_b.join()

    #stop Progress
    progress.stop()
    
    #Salvataggio data
    date = datetime.datetime.now()
    date_str = date.strftime("%d-%m-%Y-%H-%M")

    # plot delle loss
    plt.plot(epoch_history,loss_history_a, label="Loss with Normal Images")
    plt.plot(epoch_history,loss_history_b, label="Loss with Transformed Images")
    plt.title('Train Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_graphic/loss_plot_'+date_str+'.png')
    plt.show()