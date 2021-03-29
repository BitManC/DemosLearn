import torch
from torch.autograd import Variable

# part 1
#x = Variable(torch.ones(2, 2),
# requires_grad=True)

x = torch.ones(2, 2, requires_grad=True)

x.register_hook(lambda grad:grad*2)

y = x + 2
z = y * y * 3
# z = torch.sum(z)
# nn = torch.rand(2, 2)
nn = torch.ones(2, 2)
print(nn)


z.backward(gradient=nn, retain_graph=True)
torch.autograd.backward(z,
                        grad_tensors=nn,
                        retain_graph=True)

print(torch.autograd.grad(z, [x, y, z],
                    grad_outputs=nn))

print(x.grad)
print(y.grad)
print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)

#


