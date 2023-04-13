import torch
from torch import  nn
#create regression model class

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

class LinearRegressionModel(nn.Module):
    def __int__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(53)

model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(x_test)

print(f"Number of testing samples: {len(x_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")