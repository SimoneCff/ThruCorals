# Assuming you have already created a dataloader called 'dataloader' with images and predicted labels
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

# Create a function to visualize random images with predicted labels
def visualize_random_images(dataloader, model, num_images=4):
   # Set the model to evaluation mode and move it to the GPU (if available)
    model.eval()
    
    if torch.cuda.is_available():
     model.cuda()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    # Iterate over random samples from the dataloader
    for i in range(num_images):
        # Get a random sample from the dataloader
        sample = random.choice(dataloader.dataset)

        # Extract the image and label
        image, label = sample

        # Move the image tensor to the GPU (if available)
        if torch.cuda.is_available():
            image = image.cuda()

        # Make sure the image is in the correct format (batch size of 1)
        image = image.unsqueeze(0)

        # Forward pass to get predicted label
        with torch.no_grad():
            output = model(image)

        # Convert the output to a predicted label (assuming it's a classification task)
        _, predicted_label = torch.max(output, 1)

        p_label = coral_dataset.class_name[predicted_label.item()]

        # Normalize the image data to [0, 1]
        image = image.squeeze().cpu().permute(1, 2, 0)  # Reshape image tensor for plotting
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        
        # Plot the image with the true label and predicted label
        axes[i].imshow(image)  # Reshape image tensor for plotting
        axes[i].set_title(f'Predicted Label: {p_label}')
        
    date = datetime.datetime.now()
    date_str = date.strftime("%d-%m-%Y-%H-%M")
    # Show the plot
    create_dir('graphic/' + date_str)
    plt.tight_layout()
    plt.savefig('graphic/' + date_str + '/image_label.png')

# Transform Function
transform = transforms.Compose([
        transforms.Resize(rsize),  # Ridimensiona l'immagine a 256x256 pixel
        transforms.ToTensor(),  # Converti l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza i valori dei pixel
    ])
    # Dataset
coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',scale=0.2,classes='data/classes.csv',transform=transform)
coral_loader = DataLoader(dataset=coral_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_count())

    # CNN
model = CoralCNN(num_classes=len(coral_dataset.class_name))


# Call the function to visualize random images
if not torch.cuda.is_available():
        checkpoint = torch.load('checkpoint.pth',map_location=torch.device('cpu'))
else:
        checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

visualize_random_images(coral_loader, model)
