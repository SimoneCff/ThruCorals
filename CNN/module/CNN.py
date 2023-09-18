import torch.nn as nn

#Definition of the Convolutional Neural Network
class CoralCNN(nn.Module):

    def __init__(self, num_classes):
        super(CoralCNN, self).__init__()

        #First Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=20)

        #Second Layer
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Third Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3= nn.MaxPool2d(kernel_size=2, stride=2)

        #Fourth Layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #First Layer full connected
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        #Second Layer full connected
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)     

    # Progresses data across layers
    def forward(self, x):
        #FL
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        #SL
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        #TL
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        #FL
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        #Reshape e Resize input FL
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out