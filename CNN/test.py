import os
import torch
import torchvision.transforms as transforms
from module.CDS import CoralDataset
from module.CNN import CoralCNN
from torch.utils.data import DataLoader
from rich.progress import Progress
from multiprocessing import cpu_count
import delete


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

delete.delete()
batch_size =  256
num_epochs = 25
learning_rate = 0.001
rsize = (244, 244)

def Start_test():
     # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    # Transform Function
    transform = transforms.Compose([
        transforms.Resize(rsize),  # Ridimensiona l'immagine a 256x256 pixel
        transforms.ToTensor(),  # Converti l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza i valori dei pixel
    ])
    
    # Dataset
    coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',classes='data/classes.csv',scale=0.2,transform=transform)
    coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_count())

    # CNN
    model = CoralCNN(num_classes=len(coral_dataset.class_name))

    # Move model to the appropriate device
    model.to(device)

    if not torch.cuda.is_available():
        checkpoint = torch.load('checkpoint.pth',map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # Test the model
    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []

    with torch.no_grad() and Progress() as progress:
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

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            # Update progress bar
            progress.update(task, advance=1, description=f"Batch {batch_idx}/{len(coral_loader)}")  
            

        progress.update(task, completed=len(coral_loader))
        accuracy = 100 * correct / total
        print(f"Model Accuracy = {accuracy}")


if __name__ == '__main__':
    Start_test()