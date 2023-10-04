import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet, BasicBlock


class VGG(nn.Module):
    def __init__(self, dataset):
        super(VGG, self).__init__()
        if dataset == "mnist":
            self.in_channels = 1
            self.num_classes = 10
        elif dataset == "cifar10":
            self.in_channels = 3
            self.num_classes = 10

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        #         self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        #         self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #         self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        #         self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #         self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        #         self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        #         x = self.maxpool(x)
        #         x = F.relu(self.conv8(x))
        #         x = F.relu(self.conv9(x))
        #         x = F.relu(self.conv10(x))
        #         x = self.maxpool(x)
        #         x = F.relu(self.conv11(x))
        #         x = F.relu(self.conv12(x))
        #         x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5)  # dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


class CNN(nn.Module):
    def __init__(self, dataset):
        super(CNN, self).__init__()
        if dataset == "mnist":
            self.in_channels = 1
            self.num_classes = 10
            self.out = nn.Linear(32 * 7 * 7, self.num_classes)
        elif dataset == "cifar10":
            self.in_channels = 3
            self.num_classes = 10
            self.out = nn.Linear(32 * 8 * 8, self.num_classes)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def forward_once_unlearn(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def create_model(model_name, dataset):
    if model_name == 'simple_cnn':
        return CNN(dataset)
    elif model_name == 'vgg':
        return VGG(dataset)
    elif model_name == 'resnet':  # resnet 34
        return ResNet(BasicBlock, [3, 4, 6, 3], dataset)