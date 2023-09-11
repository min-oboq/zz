import torch
import copy
import glob
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

data_path = r"C:\Users\admin\zz\pythorch\data\catanddog\train"

transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    data_path,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


print(len(train_dataset))

# samples, labels = train_loader._get_iterator()._next_data()
# classes = {0: "cat", 1: "dog"}
# fig = plt.figure(figsize=(16, 24))
# for i in range(24):
#     fig.add_subplot(4, 6, i + 1)
#     plt.title(classes[labels[i].item()])
#     plt.axis("off")
#     plt.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
# plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
# plt.show()

resnet18 = models.resnet18(pretrained=True)


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


set_parameter_requires_grad(resnet18)
resnet18.fc = nn.Linear(512, 1)

for name, param in resnet18.named_parameters():
    print(name)
print("requires_grad 가 True 인 layer:-----------")
for name, param in resnet18.named_parameters():
    if param.requires_grad:
        print(name)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(512, 2)
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters())
cost = torch.nn.CrossEntropyLoss()
print(model)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=13, is_train=True):
    since = time.time()
    acc_history = []
    loss_history = []
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, lables in dataloaders:
            inputs = inputs.to(device)
            lables = lables.to(device)
            
            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,lables)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == lables.data)
    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.double() / len(dataloaders.dataset)
    
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    if epoch_acc>best_acc:
        best_acc = epoch_acc
    
    acc_history.append(epoch_acc.item)
    
    loss_history.append(epoch_loss)
    torch.save(model.state_dict(), os.path.join(r"C:\Users\admin\zz\pythorch\data\catanddog\train",'{0:0=2d}.pth'.format(epoch)))
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    return acc_history, loss_history

params_to_update = []
for name, param in resnet18.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)
        
optimizer = optim.Adam(params_to_update)