import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n):
        super(ResNet, self).__init__()
        self.n = n

        self.input_mod = nn.Sequential()
        self.input_mod.add_module('input_conv', nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.input_mod.add_module('input_conv_bn', nn.BatchNorm2d(16))
        self.input_mod.add_module('input_relu', nn.ReLU())

        '''
        self.conv32_1 = nn.Sequential()
        self.conv32_1.add_module('conv32_1_conv1', nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
        '''
        # two layers as one module
        self.conv32 = []
        for i in range(self.n):
            self.conv32.append(nn.Sequential())
            self.conv32[i].add_module('conv32_conv' + str(i) + '_0', nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.conv32[i].add_module(
                'conv32_bn' + str(i) + '_0', nn.BatchNorm2d(16))
            self.conv32[i].add_module('conv32_relu' + str(i) + '_0', nn.ReLU())

            self.conv32[i].add_module('conv32_conv' + str(i) + '_1', nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.conv32[i].add_module(
                'conv32_bn' + str(i) + '_1', nn.BatchNorm2d(16))

        self.conv32 = nn.ModuleList(self.conv32)

        # down sampling from 32x32 to 16x16 and the following conv layer
        self.down_sample_32_to_16 = nn.Sequential()
        self.down_sample_32_to_16.add_module('down_sample_32_to_16_conv_0', nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
        self.down_sample_32_to_16.add_module(
            'down_sample_32_to_16_bn_0', nn.BatchNorm2d(32))
        self.down_sample_32_to_16.add_module(
            'down_sample_32_to_16_relu_0', nn.ReLU())

        self.down_sample_32_to_16.add_module('down_sample_32_to_16_conv_1', nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.down_sample_32_to_16.add_module(
            'down_sample_32_to_16_bn_1', nn.BatchNorm2d(32))
        self.down_sample_32_to_16.add_module(
            'down_sample_32_to_16_relu_1', nn.ReLU())

        self.conv16 = []
        for i in range(self.n - 1):
            self.conv16.append(nn.Sequential())
            self.conv16[i].add_module('conv16_conv' + str(i) + '_0', nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.conv16[i].add_module(
                'conv16_bn' + str(i) + '_0', nn.BatchNorm2d(32))
            self.conv16[i].add_module('conv16_relu' + str(i) + '_0', nn.ReLU())

            self.conv16[i].add_module('conv16_conv' + str(i) + '_1', nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.conv16[i].add_module(
                'conv16_bn' + str(i) + '_1', nn.BatchNorm2d(32))

        self.conv16 = nn.ModuleList(self.conv16)

        # down sampling from 16x16 to 8x8 and the following conv layer
        self.down_sample_16_to_8 = nn.Sequential()
        self.down_sample_16_to_8.add_module('down_sample_16_to_8_conv_0', nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.down_sample_16_to_8.add_module(
            'down_sample_16_to_8_bn_0', nn.BatchNorm2d(64))
        self.down_sample_16_to_8.add_module(
            'down_sample_16_to_8_relu_0', nn.ReLU())

        self.down_sample_16_to_8.add_module('down_sample_16_to_8_conv_1', nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.down_sample_16_to_8.add_module(
            'down_sample_16_to_8_bn_1', nn.BatchNorm2d(64))
        self.down_sample_16_to_8.add_module(
            'down_sample_16_to_8_relu_1', nn.ReLU())

        self.conv8 = []
        for i in range(self.n - 1):
            self.conv16.append(nn.Sequential())
            self.conv16[i].add_module('conv16_conv' + str(i) + '_0', nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.conv16[i].add_module(
                'conv16_bn' + str(i) + '_0', nn.BatchNorm2d(32))
            self.conv16[i].add_module('conv16_relu' + str(i) + '_0', nn.ReLU())

            self.conv16[i].add_module('conv16_conv' + str(i) + '_1', nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.conv16[i].add_module(
                'conv16_bn' + str(i) + '_1', nn.BatchNorm2d(32))

        self.conv8 = nn.ModuleList(self.conv8)

        self.avg_pool = nn.AvgPool2d(2, stride=1, padding=0)
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        res = self.input_mod(x)
        for mod in self.conv32:
            temp = mod(res)
            temp = temp + res
            temp = F.relu(temp)
            res = temp
        res = self.down_sample_32_to_16(res)
        for mod in self.conv16:
            temp = mod(res)
            temp = temp + res
            temp = F.relu(temp)
            res = temp
        res = self.down_sample_16_to_8(res)
        for mod in self.conv8:
            temp = mod(res)
            temp = temp + res
            temp = F.relu(temp)
            res = temp
        x = self.avg_pool(res)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc(x)
        return x
