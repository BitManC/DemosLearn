import torch

dev = torch.device("cpu")
dev = torch.device("cuda")
a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev).to_dense()
print(a)






