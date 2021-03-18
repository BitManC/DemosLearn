import torch

a = torch.zeros((2, 4))
b = torch.ones((2, 4))

out = torch.cat((a,b),dim=0)
print(out)

out = torch.cat((a,b),dim=1)
print(out)

#torch.stack

print("torch.stack")
a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
print(a, b)
out = torch.stack((a, b), dim=2)
print(out)
print(out.shape)

print(out[:, :, 0])
print(out[:, :, 1])









