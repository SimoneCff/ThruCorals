import torch
from torch import nn
import matplotlib.pyplot as plt

#parametri noti
weight = 0.7
bias = 0.3

#data
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y= weight * x + bias

train_split = int(0.8*len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test= x[train_split:], y[train_split:]

def plot_pred(train_data=x_train,
              train_labels=y_train,
              test_data=x_test,
              test_labels=y_test,
              predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels, c="b", s=4, label="Train data")
    plt.scatter(test_data,test_labels, c="r", s=4, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(prop={"size":14})
    plt.show()

plot_pred()