import torch.nn as nn
import torch.nn.functional as F


class PlainNet(nn.Module):
    def __init__(self, n):
        super(PlainNet, self).__init__()
        self.n = n

        self.model = nn.Sequential()
        self.model.add_module('input_conv', nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.model.add_module('input_conv_bn', nn.BatchNorm2d(16))
        self.model.add_module('input_relu', nn.ReLU())

        for i in range(2 * self.n - 1):
            self.model.add_module('conv_series0_' + str(i), nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.model.add_module(
                'conv_series0_' + str(i) + 'bn', nn.BatchNorm2d(16))
            self.model.add_module('conv_series0_' + str(i) + 'relu', nn.ReLU())

        self.model.add_module('down_sample_0', nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
        self.model.add_module(
            'down_sample_0_bn', nn.BatchNorm2d(32))
        self.model.add_module('down_sample_0_relu', nn.ReLU())

        for i in range(2 * self.n - 1):
            self.model.add_module('conv_series1_' + str(i), nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.model.add_module(
                'conv_series1_' + str(i) + 'bn', nn.BatchNorm2d(32))
            self.model.add_module('conv_series1_' + str(i) + 'relu', nn.ReLU())

        self.model.add_module('down_sample_1', nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.model.add_module(
            'down_sample_1_bn', nn.BatchNorm2d(64))
        self.model.add_module('down_sample_1_relu', nn.ReLU())

        for i in range(2 * i):
            self.model.add_module('conv_series2_' + str(i), nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            self.model.add_module(
                'conv_series2_' + str(i) + 'bn', nn.BatchNorm2d(64))
            self.model.add_module('conv_series2_' + str(i) + 'relu', nn.ReLU())

        self.model.add_module('avg_pool', nn.AvgPool2d(2, stride=1, padding=0))

        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc(x)
        return x
