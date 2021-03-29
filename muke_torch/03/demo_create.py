import torch
a = torch.Tensor([[1, 2],[3, 4]])
print(a)

b = torch.Tensor(2, 2)
print(b)

d = torch.tensor(((1, 2), (3, 4)))
print(d.type())
print(d.type_as(a))

d = torch.empty(2,3)
print(d.type())
print(d.type_as(a))

d = torch.zeros(2,3)
print(d.type())
print(d.type_as(a))

d = torch.zeros_like(d)
print(d.type())
print(d.type_as(a))

d = torch.eye(2, 2)
print(d.type())
print(d.type_as(a))

d = torch.ones(2, 2)
print(d.type())
print(d.type_as(a))

d = torch.ones_like(d)
print(d.type())
print(d.type_as(a))

d = torch.rand(2, 3)
print(d.type())
print(d.type_as(a))

d = torch.arange(2, 10, 2)
print(d.type())
print(d.type_as(a))

d = torch.linspace(10, 2, 3)
print(d.type())
print(d.type_as(a))

dd = torch.normal(mean=0, std=1, size=(2, 3), out=b)
print(b)
print(dd)

d = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(d.type())
print(d.type_as(a))

d = torch.Tensor(2, 2).uniform_(-1, 1)
print(d.type())
print(d.type_as(a))


d = torch.randperm(10)
print(d.type())
print(d.type_as(a))
