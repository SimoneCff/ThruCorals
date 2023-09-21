import torchvision.transforms as transforms
from module.CDS import CoralDataset

def test():
    rsize = (244, 244)
    transform = transforms.Compose([
        transforms.Resize(rsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    coral_dataset = CoralDataset(csv= 'data/combined_annotations.csv',data_dir='data/images/images',scale=0.2,transform=transform, classes='data/classes.csv')
    print(coral_dataset.class_name)

if __name__ == '__main__':
    test()
