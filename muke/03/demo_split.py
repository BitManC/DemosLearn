import torch

a = torch.rand((3, 4))
print(a)
out = torch.chunk(a, 2, dim=1)
print(out[0], out[0].shape)
print(out[1], out[1].shape)


a = torch.rand((10, 4))
print(a)
out = torch.split(a, 3, dim=0)
print(len(out))
for t in out:
    print(t, t.shape)

out = torch.split(a, [1, 3, 6], dim=0)
for t in out:
    print(t, t.shape)


