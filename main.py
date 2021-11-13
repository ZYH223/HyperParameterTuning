import math
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np


def load_cifar10(trn_val_capacity=-1, tst_capacity=-1):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    full_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    if trn_val_capacity < 0 or trn_val_capacity > len(full_set):
        trn_val_capacity = len(full_set)
    trn_size = int(0.9 * trn_val_capacity)
    val_size = len(full_set) - trn_size
    _train_set = torch.utils.data.Subset(full_set, range(0, trn_size))
    _validate_set = torch.utils.data.Subset(full_set, range(trn_size, trn_size + val_size))
    # _train_set, _validate_set = torch.utils.data.random_split(full_set, [trn_size, val_size])

    _test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
    if 0 < tst_capacity < len(_test_set):
        # _test_set = torch.utils.data.random_split(_test_set, [tst_capacity])
        _test_set = torch.utils.data.Subset(_test_set, range(0, tst_capacity))
    _classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return _train_set, _validate_set, _test_set, _classes


def make_dataloader(dataset, _batch_size, shuffle=True):
    batch_number = math.ceil(len(dataset) / _batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size,
                                             shuffle=shuffle, num_workers=1,
                                             drop_last=True)
    return dataloader, batch_number


# functions to show an image
def show_img(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def count_time(name, last_time):
    new_time = time.time()
    print(f'{name} complete in {new_time - last_time}s')
    last_time = new_time
    return new_time


class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()
        self.hidden1 = nn.Linear(3 * 32 * 32, 32 * 32)
        self.hidden2 = nn.Linear(32 * 32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, z):
        z = functional.relu(self.hidden1(z))
        z = torch.sigmoid(self.hidden2(z))
        z = self.output(z)
        return z


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.in_channels = 3
        self.conv3_16 = self.__make_layer(16, 2)
        self.conv3_32 = self.__make_layer(32, 2)
        self.conv3_64 = self.__make_layer(64, 3)
        self.conv3_128a = self.__make_layer(128, 3)
        self.conv3_128b = self.__make_layer(128, 3)
        self.fc1 = nn.Linear(1 * 1 * 128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv3_16(x)
        x = functional.max_pool2d(x, 2)
        x = self.conv3_32(x)
        x = functional.max_pool2d(x, 2)
        x = self.conv3_64(x)
        x = functional.max_pool2d(x, 2)
        x = self.conv3_128a(x)
        x = functional.max_pool2d(x, 2)
        x = self.conv3_128b(x)
        x = functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = functional.relu(x)
        return functional.softmax(self.fc3(x), dim=0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print('using device ' + str(device))

    # batch_size_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    batch_size_list = [4]
    for batch_size in batch_size_list:
        # 1st Step: Load and process dataset
        # loss_iter_cap = math.ceil((50000.0 / batch_size) / 5)
        plot_epoch = []
        plot_accuracy = []
        plot_time = []
        epoch_size = 200
        print('Load data with batch size of ' + str(batch_size))
        train_set, validate_set, test_set, classes = load_cifar10()
        train_size = len(train_set)
        validate_size = len(validate_set)
        test_size = len(test_set)
        train_loader, _ = make_dataloader(train_set, batch_size)
        validate_loader, _ = make_dataloader(validate_set, batch_size)
        test_loader, _ = make_dataloader(test_set, batch_size)

        # 2nd Step: Initialize
        net = LeNet5().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(net.parameters())

        # 3rd Step: Training NN
        print('epoch    accuracy    time    learning_rate')
        train_start = time.time()
        for epoch in range(epoch_size):
            # Train
            # running_loss = 0.0
            epoch_start = time.time()
            net.train()
            # cur_time = epoch_start
            for i, data in enumerate(train_loader):
                inputs, labels = data
                # inputs = inputs.reshape(inputs.shape[0],
                #                         inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs.shape)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # cur_time = count_time('batch compute', cur_time)

                # running_loss += loss.item()
                # if i % loss_iter_cap == loss_iter_cap-1 or i == train_size - 1:
                #     print('[%d, %5d*%4d] loss: %.3f' %
                #           (epoch + 1, i + 1, batch_size, running_loss / loss_iter_cap))
                #     running_loss = 0.0

            # Validate
            with torch.no_grad():
                net.eval()
                correct = 0
                for i, data in enumerate(validate_loader):
                    inputs, labels = data
                    # inputs = inputs.reshape(inputs.shape[0],
                    #                         inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                accuracy = correct / validate_size
            print(f'{epoch + 1:<5d}\t'
                  f'{accuracy:2.2%}\t\t'
                  f'{time.time() - epoch_start:.2f}s\t'
                  f'{optimizer.param_groups[0]["lr"]:f}')
            plot_epoch.append(epoch + 1)
            plot_accuracy.append(accuracy)
            plot_time.append(time.time() - epoch_start)
            # print('epoch%-4d Accuracy on validate set: %d %%' % (
            #     epoch + 1, 100 * accuracy))
            # print(f'Total time consume:{time.time() - epoch_start}s')

        torch.cuda.synchronize()
        train_end = time.time()
        print(f'Training finished with {sum(plot_time):.2f}s({sum(plot_time)/len(plot_time):.2f}s on average)')

        # 4th Step: Validate the results and performance
        with torch.no_grad():
            net.eval()
            correct = 0
            for i, data in enumerate(test_loader):
                inputs, labels = data
                # inputs = inputs.reshape(inputs.shape[0],
                #                         inputs.shape[1] * inputs.shape[2] * inputs.shape[3])
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs.shape)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            accuracy = correct / test_size
        print('Accuracy of the network on the %d test images: %d %%' % (
            test_size, 100 * accuracy))

        fig, ax = plt.subplots()
        ax.plot(plot_epoch, plot_accuracy)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        plt.ylim(0, 1.0)
        ax.set(xlabel='epoch', ylabel='accuracy',
               title='Training accuracy on validate set')
        ax.grid()
        fig.savefig("Training accuracy.png")
        # plt.show()

    # 5th Step(Optional): Save the model and statistical data
    # torch.save(net.state_dict(), './model/ResNet18.pt')
    # torch.save(net.state_dict(), './model/FC2.pt')
    torch.save(net.state_dict(), './model/model.pt')
