import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
#data
train_data = dataset.MNIST(root="mnist",
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_data = dataset.MNIST(root="mnist",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)
#batchsize
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)

cnn = CNN()
cnn = cnn.cuda()
#loss

loss_func = torch.nn.CrossEntropyLoss()

#optimizer

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

#training
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {}, ite is "
          "{}/{}, loss is {}".format(epoch+1, i,
                                     len(train_data) // 64,
                                     loss.item()))
    #eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        #[batchsize]
        #outputs = batchsize * cls_num
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print("epoch is {}, accuracy is {}, "
          "loss test is {}".format(epoch + 1,
                                   accuracy,
                                   loss_test.item()))

torch.save(cnn, "model/mnist_model.pkl")





