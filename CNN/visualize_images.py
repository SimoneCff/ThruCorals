
import matplotlib.pyplot as plt
import random
import torch
import os
from torch.utils.data import DataLoader
from module.CDS import CoralDataset
from module.CNN import CoralCNN
from multiprocessing import cpu_count
import torchvision.transforms as transforms
import datetime

batch_size =  256
rsize = (244, 244)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_random_images(dataloader, model, num_images=4):
    model.eval()
    
    if torch.cuda.is_available():
     model.cuda()

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        sample = random.choice(dataloader.dataset)
        image, label = sample

        if torch.cuda.is_available():
            image = image.cuda()

        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)

        _, predicted_label = torch.max(output, 1)
        p_label = coral_dataset.class_name[predicted_label.item()]

        image = image.squeeze().cpu().permute(1, 2, 0)  # Reshape image tensor for plotting
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        
        axes[i].imshow(image)
        axes[i].set_title(f'Predicted Label: {p_label}')
        
    date = datetime.datetime.now()
    date_str = date.strftime("%d-%m-%Y-%H-%M")
    create_dir('graphic/' + date_str)
    plt.tight_layout()
    plt.savefig('graphic/' + date_str + '/image_label.png')

transform = transforms.Compose([
        transforms.Resize(rsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',scale=0.2,classes='data/classes.csv',transform=transform)
coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_count())
model = CoralCNN(num_classes=len(coral_dataset.class_name))

if not torch.cuda.is_available():
        checkpoint = torch.load('checkpoint.pth',map_location=torch.device('cpu'))
else:
        checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

visualize_random_images(coral_loader, model)
