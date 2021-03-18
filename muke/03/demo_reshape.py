import torch

a = torch.rand(2, 3)

print(a)

out = torch.reshape(a, (3, 2))

print(out)

print(torch.t(out))

print(torch.transpose(out, 0, 1))

a = torch.rand(1, 2, 3)
out = torch.transpose(a, 0, 1)
print(out)
print(out.shape)

out = torch.squeeze(a)
print(out)
print(out.shape)
out = torch.unsqueeze(a, -1)
print(out.shape)

out = torch.unbind(a, dim=2)

print(out)

print(a)
print(torch.flip(a, dims=[2, 1]))

print(a)
out = torch.rot90(a, -1, dims=[0, 2])
print(out)
print(out.shape)