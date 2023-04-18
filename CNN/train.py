import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import CNN
import torch.nn as nn

RESIZE_IMG = (128, 128)
num_classes = 10
batch_size = 64
learining_rate = 0.001
num_epochs = 10

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    #Add Arguments
    parser.add_argument('--size', default=128, help='Resize Dataset images for training')
    par

def Start_Train():
    # Check CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.Resize(RESIZE_IMG),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])
    #Load CNN
    model = CNN.ConvNeuralNet()

    #init Dataset (Same for train and test)

    train_dt = torchvision.datasets.SEADT(root='./data',
                                          train=True,
                                          transform=transform)


    test_dt = torchvision.datasets.SEADT(root='./data',
                                          train=False,
                                          transform=transform)

    #Load Dataset
    train_dl = torch.utils.data.DataLoader(dataset=train_dt,
                                           batch_size=batch_size,
                                           shuffle=True)

    test_dl = torch.utils.data.DataLoader(dataset=test_dt,
                                          batch_size=batch_size,
                                          shuffle=True)

    #Initi LossFunction
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learining_rate, momentum= 0.9)


    #Start Loop Train
    for epoch in range(num_epochs):
        # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))