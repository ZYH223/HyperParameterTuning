from abc import ABC

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def load_cifar10(batch_size=4, shuffle=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    _train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    _test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=4)
    _classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return _train_loader, _test_loader, _classes


# functions to show an image
def show_img(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # TODO
    # Epoch BatchSize Tuning
    # Regularization & Random Dropout
    # Loss Function
    # Optimizer

    # 1st Step: Load and process dataset
    train_loader, test_loader, classes = load_cifar10()

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show_img(torchvision.utils.make_grid(images))

    # 2nd Step: Initialize
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 3rd Step: Training NN
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Training Finished')

    # 4th Step: Validate the results and performance
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * accuracy))
    # 5th Step(Optional): Save the model to disk
