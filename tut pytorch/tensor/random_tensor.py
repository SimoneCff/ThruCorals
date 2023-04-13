import torch
import random

RANDOM_SEED= 54
torch.manual_seed(seed=RANDOM_SEED)

random_tensor_A= torch.rand(3, 4)
random_tensor_B= torch.rand(3, 4)

print(f"Tensor A: \n{random_tensor_A}\n")
print(f"Tensor B: \n{random_tensor_B}\n")

print("Equal? : \n")
print(random_tensor_B == random_tensor_A)

