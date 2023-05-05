import torch
import torchvision.transforms as transforms
from PIL import Image

# Carica l'immagine
img_path = 'path/to/image.jpg'
img = Image.open(img_path).convert('RGB')

# Preprocessa l'immagine
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# Carica la CNN
model = SeaCNN(num_classes=10)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Esegui l'inferenza
model.eval()
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(probs, 1)

# Stampa l'output previsto
print('Classe prevista:', predicted.item())
