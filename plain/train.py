import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import PlainNet

path = "./models/56/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_crop = transforms.RandomCrop(32)
h_flip = transforms.RandomHorizontalFlip()


transform = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


net = PlainNet(n=9)
net = net.to(device=device)
net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0001)

iteration_count = 0
checkpoint_steps = 10000
loss_list = []
total_iteration = 64000

optimizer.lr = 0.1
running_loss = 0.0
for epoch in range(1000000):
    for i, data in enumerate(trainloader, 0):
        iteration_count += 1
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if iteration_count == 100:
            optimizer.lr = 0.01

        if iteration_count == 48000:
            optimizer.lr = 0.001

        if iteration_count % 100 == 0:
            print('[%d, %d] loss: %f' %
                  (epoch + 1, iteration_count, running_loss / 100))
            loss_list.append(running_loss / 100)
            running_loss = 0.0

        if (iteration_count % checkpoint_steps) == 0:
            model_save_path = path + str(iteration_count) + ".model"
            torch.save(net.state_dict(), model_save_path)

        if iteration_count == total_iteration:
            break
    if iteration_count == total_iteration:
        break

np.savetxt(path + "loss.log",
           np.array(loss_list), delimiter="\n")

model_save_path = path + str(iteration_count) + ".model"
torch.save(net.state_dict(), model_save_path)


print('Finished Trainning')
