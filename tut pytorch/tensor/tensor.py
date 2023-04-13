import torch


def scalar(size):
    scalar = torch.tensor(size)
    print("Scalar tensor :")
    #general print
    print(scalar)
    #print dimension
    print(scalar.ndim)
    #print items insidde
    print(scalar.item())
    return

def vector():
    vector = torch.tensor([7, 7])
    print("vector tensor :")
    #general print
    print(vector)
    #print dimension
    print(vector.ndim)
    #print shape inside
    print(vector.shape)
    return

def tensor():
    tensor = torch.tensor([[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]])
    print("Tensor :")
    print(tensor)


scalar(7)
vector()