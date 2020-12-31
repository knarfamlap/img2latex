import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):

    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3)

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv5 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv6 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3)

    def forward(self, img):
        img = img.type(torch.FloatTensor) / 255.

        img = self.max_pool1(F.relu(self.conv1(img)))
        img = self.max_pool2(F.relu(self.conv2(img)))
        img = F.relu(self.conv3(img))

        img = self.max_pool3(F.relu(self.conv4(img)))
        img = self.max_pool4(F.relu(self.conv5(img)))

        img = F.relu(self.conv6(img))

        return img
