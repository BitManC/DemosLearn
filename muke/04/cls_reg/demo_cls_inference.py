import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
#net

test_data = dataset.MNIST(root="mnist",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

cnn = torch.load("model/mnist_model.pkl")
cnn = cnn.cuda()
#loss
#eval/test
loss_test = 0
accuracy = 0

import cv2


#pip install opencv-python -i http://mirrors.aliyun.com/pypi/simple/   --trusted-host mirrors.aliyun.com
for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    #batchsize * 1 * 28 * 28

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)
accuracy = accuracy / len(test_data)
print(accuracy)








