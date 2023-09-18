import os
import delete
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from module.CDS import CoralDataset
from module.CNN import CoralCNN
from torch.utils.data import DataLoader
from rich.progress import Progress
from multiprocessing import cpu_count


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

delete.delete()
batch_size =  256
num_epochs = 20
learning_rate = 0.001
rsize = (244, 244)

def Start_Train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize(rsize),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',scale=0.2,classes='data/classes.csv',transform=transform)
    coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_count())

    model = CoralCNN(num_classes=len(coral_dataset.class_name))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_history = []
    accuarcy_history = []
    epoch_history = []

    # Train loop
    with Progress() as progress:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            task = progress.add_task("Training Model", total=len(coral_loader))
            model.train()

            for batch_idx, (images, labels) in enumerate(coral_loader):

                images = images.to(device)
                labels = torch.as_tensor(labels).to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                progress.update(task, advance=1, description=f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx}/{len(coral_loader)}", status=f"Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(coral_dataset)
            loss_history.append(epoch_loss)
            epoch_history.append((epoch + 1)+36)

            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")

            model.eval()

            correct = 0
            total = 0
            task = progress.add_task("Testing Model", total=len(coral_loader))

            for batch_idx, (images, labels) in enumerate(coral_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                progress.update(task, advance=1, description=f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_idx}/{len(coral_loader)}")  
            progress.update(task, completed=len(coral_loader))
            accuracy = 100 * correct / total
            accuarcy_history.append(accuracy)
            print(f"Epoch {epoch + 1}/{num_epochs} Model Accuracy = {accuracy}")
            

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoint.pth')

    date = datetime.datetime.now()
    date_str = date.strftime("%d-%m-%Y-%H-%M")

    create_dir('graphic/' + date_str)

    plt.plot(epoch_history, loss_history)
    plt.title('Train Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('graphic/' + date_str + '/loss_plot.png')

    plt.clf()
    plt.plot(epoch_history, accuarcy_history)
    plt.title('Train Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.savefig('graphic/' + date_str + '/accuracy_plot.png')

    print("Train Done, Exit")


if __name__ == '__main__':
    Start_Train()


