
import torch
import torchvision.transforms as transforms
from .module.DTS import SeaDataset
from .module.CNN import SeaCNN
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TimeElapsedColumn
from multiprocessing import  cpu_count
import threading

nclass = 3
batch_size = 256
num_workers = 16
work = cpu_count()

def testing_model(model,dataloader,checkpoint,type,progress,task,device):
     # carica i pesi dal checkpoint
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test the model

    model.eval()  # Set the model to evaluation mode1

    with torch.no_grad() and progress:
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress.update(task, advance=1)
        progress.update(task, completed=len(dataloader))
        print(f'Test Accuracy of the model {type} : {100 * correct / total} %')

def Start_test(path_in, path_in_transf):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Dataset
    test_notransform_dataset = SeaDataset(data_dir=path_in, transform=transform)
    test_transform_dataset = SeaDataset(data_dir=path_in_transf, transform=transform)

    # Data loader
    test_notransform_loader = DataLoader(dataset=test_notransform_dataset, batch_size=batch_size, shuffle=True, num_workers=work)
    test_transform_loader = DataLoader(dataset=test_transform_dataset, batch_size=batch_size, shuffle=True, num_workers=work)

    # Model
    model_a = SeaCNN(num_classes=nclass).to(device)
    model_b = SeaCNN(num_classes=nclass).to(device)

    #Creazione Progress Bar
    progress = Progress("[progress.description]{task.description}", BarColumn(), TextColumn("[bold green]{task.completed}/{task.total}"),"|",TimeElapsedColumn() ,"||",TimeRemainingColumn(),"|")

    #Creazione Task
    task_a = progress.add_task("[cyan]Testing Model A...", total=len(test_notransform_loader), task_id="a")
    task_b = progress.add_task("[magenta]Testing Model B...", total=len(test_notransform_loader), task_id="b")

    #Creazione Threads
    thread_a = threading.Thread(target=testing_model, args=(model_a,test_notransform_loader,"checkpoint_a.pth","A",progress,task_a,device))
    thread_b = threading.Thread(target=testing_model, args=(model_b,test_notransform_loader,"checkpoint_b.pth","B",progress,task_b,device))
    
    #Start Threads
    thread_a.start()
    thread_b.start()

    #Riunificazione Threads
    thread_a.join()
    thread_b.join()

    #stop Progress
    progress.stop() 

if __name__ == '__main__':
    Start_test()