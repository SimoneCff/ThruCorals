import torchvision.transforms as transforms
from module.CDS import CoralDataset

def test():
    rsize = (244, 244)
    # Transform Function
    transform = transforms.Compose([
        transforms.Resize(rsize),  # Ridimensiona l'immagine a 256x256 pixel
        transforms.ToTensor(),  # Converti l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza i valori dei pixel
    ])
    
    # Dataset
    coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',scale=0.2,transform=transform, classes='data/classes.csv')

    print(coral_dataset.class_name)

if __name__ == '__main__':
    test()