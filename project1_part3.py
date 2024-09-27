import argparse
import logging
import sys
import time
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ==================================
# 控制输入选项
def parse_args():
    parser = argparse.ArgumentParser(description='script for part 3 of project 1')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
                        help='The path that stores the log files.')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='When using this option, only run the test functions.')
    pargs = parser.parse_args()
    return pargs

# 创建日志
def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter) 
    clogger.addHandler(ch)
    return clogger

# 训练过程
def train_net(net, trainloader, logging, criterion, optimizer, scheduler, epochs=1):
    net = net.train()
    for epoch in range(epochs):  # 运行epoch
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.cuda:
                loss = loss.cpu()

            running_loss += loss.item()
            if i % 2000 == 1999:    # 每2000个mini-batches打印一次
                logging.info('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # save network
    # torch.save(net.state_dict(), args.output_path + 'modified.pth')
    # write finish to the flie
    logging.info('Finished Training')

# 评估过程
def eval_net(net, loader, logging, mode="baseline"):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    if args.pretrained:
        if args.cuda:
            net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cuda'))
        else:
            net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logging.info('=' * 55)
    logging.info('SUMMARY of ' + mode)
    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    logging.info('=' * 55)


# 准备日志和设置GPU
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# ==================================

################################################################################################################################################

# Transformation definition
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪到32x32，padding用于确保裁剪区域总是存在
    transforms.Resize((32,32)),# Resize #略显多余
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = transforms.ToTensor()

################################################################################################################################################

# Define training and test dataset

#加载训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

#加载测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

################################################################################################################################################
# Define your baseline network
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

################################################################################################################################################

################################################################################################################################################
# Define your modified network
class Modified(nn.Module):
    def __init__(self):
        super(Modified, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

################################################################################################################################################

# ==================================
# 使用cuda如果调用了'--cuda'
baseline = Baseline()
modified = Modified()
#if args.cuda:      不启用gpu，因为我电脑上没有gpu
#    baseline = baseline.cuda()
#    modified = modified.cuda()
# ==================================

################################################################################################################################################
# Define Optimizer or Scheduler

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(modified.parameters(), lr=0.0005, momentum=0.9)

# 学习率调度器，因为暂时只有20个epoch来进行训练，模型没有太多时间来学习和收敛。在这种情况下，使用StepLR调度器可能不是最佳选择，因为它通常用于跨越数十甚至数百个epoch的训练过程，所以选择不使用
scheduler = None
################################################################################################################################################

# ==================================
# 完成训练和测试网络，并写入日志
if __name__ == '__main__':     # 这用于在Windows中运行
    # 训练modified网络
    train_net(modified, trainloader, logging, criterion, optimizer, scheduler)

    # 测试baseline网络和modified网络
    eval_net(baseline, testloader, logging, mode="baseline")
    eval_net(modified, testloader, logging, mode="modified")
# ==================================