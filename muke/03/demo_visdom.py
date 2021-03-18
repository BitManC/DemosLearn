import visdom
import torch

vis = visdom.Visdom(env="test")

x = torch.arange(1, 100, 0.01)
y = torch.sin(x)

vis.line(X=x, Y=y, win="sin", opts={'title':"y=sin(x)"})

