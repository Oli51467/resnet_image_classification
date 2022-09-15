import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import resnet34
from utils import train_and_val, plot_acc, plot_loss

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 8

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("./data/go/training/", transform=data_transform["train"])  # 训练集数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)  # 加载数据
    len_train = len(train_dataset)
    val_dataset = datasets.ImageFolder("./data/go/validation/", transform=data_transform["val"])  # 测试集数据
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4)  # 加载数据
    len_val = len(val_dataset)

    net = resnet34(3)
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 设置优化器和学习率
    epoch = 8

    history = train_and_val(epoch, net, train_loader, len_train, val_loader, len_val, loss_function, optimizer, device)

    plot_loss(np.arange(0, epoch), history)
    plot_acc(np.arange(0, epoch), history)
