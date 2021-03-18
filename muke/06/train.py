import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import ResNet18
from mobilenetv1 import mobilenetv1_small
from inceptionMolule import InceptionNetSmall
from base_resnet import resnet
from resnetV1 import resnet as resnetV1
from pre_resnet import pytorch_resnet18
from load_cifar10 import train_loader, test_loader
import os
import tensorboardX


#是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_num = 200
lr = 0.1
batch_size = 128
net = pytorch_resnet18().to(device)

#loss
loss_func = nn.CrossEntropyLoss()

#optimizer
# optimizer = torch.optim.Adam(net.parameters(), lr= lr)

optimizer = torch.optim.SGD(net.parameters(), lr = lr,
                momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=10,
                                            gamma=0.9)

model_path = "models/pytorch_resnet18"
log_path = "logs/pytorch_resnet18"
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
writer = tensorboardX.SummaryWriter(log_path)

step_n = 0
for epoch in range(epoch_num):
    print(" epoch is ", epoch)
    net.train() #train BN dropout

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).cpu().sum()
        # print("epoch is ", epoch)
        # print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
        # print("train step", i, "loss is:", loss.item(),
        #       "mini-batch correct is:", 100.0 * correct / batch_size)

        writer.add_scalar("train loss", loss.item(), global_step=step_n)
        writer.add_scalar("train correct",
                          100.0 * correct.item() / batch_size, global_step=step_n)

        im = torchvision.utils.make_grid(inputs)
        writer.add_image("train im", im, global_step=step_n)

        step_n += 1

    torch.save(net.state_dict(), "{}/{}.pth".format(model_path,
                                                     epoch + 1))
    scheduler.step()

    sum_loss = 0
    sum_correct = 0
    for i, data in enumerate(test_loader):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()

        sum_loss += loss.item()
        sum_correct += correct.item()
        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test im", im, global_step=step_n)

    test_loss = sum_loss * 1.0 / len(test_loader)
    test_correct = sum_correct * 100.0 / len(test_loader) / batch_size

    writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
    writer.add_scalar("test correct",
                      test_correct, global_step=epoch + 1)



    print("epoch is", epoch + 1, "loss is:", test_loss,
          "test correct is:", test_correct)

writer.close()























