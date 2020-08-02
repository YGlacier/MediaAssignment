import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

param_path = "./models/20/64000.model"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

net = ResNet(n=3)
net = net.to(device=device)
net = torch.nn.DataParallel(net)

net.load_state_dict(torch.load(param_path))

correct_count = 0
total_count = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, pridicted = torch.max(outputs.data, 1)
        labels = labels.cuda()
        total_count += labels.size(0)
        correct_count += (pridicted == labels).sum().item()

print("Accuracy: %f %%" % (100 * correct_count / total_count))


# 20: 82.66%
# 32: 86.41%
# 44: 86.02%
# 56: 87.03%
