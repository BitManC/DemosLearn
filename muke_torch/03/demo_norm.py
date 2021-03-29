import torch

a = torch.rand(2, 1)
b = torch.rand(2, 1)
print(a, b)
print(torch.dist(a, b, p = 1))
print(torch.dist(a, b, p = 2))
print(torch.dist(a, b, p = 3))

print(torch.norm(a))
print(torch.norm(a, p=3))
print(torch.norm(a, p='fro'))
