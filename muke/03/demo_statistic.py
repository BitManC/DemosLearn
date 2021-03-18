import torch

a = torch.rand(2, 2)

print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a, dim=0))

print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=0))

print(torch.std(a))
print(torch.var(a))

print(torch.median(a))
print(torch.mode(a))


a = torch.rand(2, 2) * 10
print(a)
print(torch.histc(a, 6, 0, 0))

a = torch.randint(0, 10, [2, 10])
print(a)
print(torch.bincount(a))

#统计某一类别样本的个数




