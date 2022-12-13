from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
import time
from collections import OrderedDict
import csv
import PIL.Image




class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        fh = open(root, 'r')
        imgs = []
        for line in fh:
            words = line.split(',')
            imgs.append((words[0],words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = PIL.Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data= MyDataset(root='/Users/lichen/Desktop/test/additional_data_labels_R.csv',transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize(64),
]))
batch_size = 1
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=False)



dronedesign_data= MyDataset(root='/Users/lichen/Desktop/test/dronedesign.csv',transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize(64),
]))

dronedesignloader = torch.utils.data.DataLoader(dronedesign_data, batch_size=batch_size,
                                         shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5),stride=1, padding=1, padding_mode='zeros', bias=False)

    def forward(self, input):
        xx1 = self.main(input)
        return xx1

D = Discriminator()

D.main.weight = nn.Parameter(torch.tensor([[[[1.,0.,0.,0.,0.],
                                [1.,0.,0.,0.,0.],
                                [0.,1.,0.,0.,0.],
                                [0.,1.,1.,0.,0.],
                                [0.,0.,0.,1.,1.]]],
                              [[[1.,1.,0.,0.,0.],
                                [0.,0.,1.,1.,0.],
                                [0.,0.,0.,1.,0.],
                                [0.,0.,0.,0.,1.],
                                [0.,0.,0.,0.,1.]]],
                              [[[0.,0.,0.,1.,1.],
                                [0.,1.,1.,0.,0.],
                                [0.,1.,0.,0.,0.],
                                [1.,0.,0.,0.,0.],
                                [1.,0.,0.,0.,0.]]],
                              [[[0.,0.,0.,0.,1.],
                                [0.,0.,0.,0.,1.],
                                [0.,0.,0.,1.,0.],
                                [0.,0.,1.,1.,0.],
                                [1.,1.,0.,0.,0.]]],
                              [[[0.,0.,1.,0.,0.],
                                [0.,1.,0.,0.,0.],
                                [0.,1.,0.,0.,0.],
                                [0.,1.,0.,0.,0.],
                                [0.,0.,1.,0.,0.]]],
                              [[[0.,0.,1.,0.,0.],
                                [0.,0.,0.,1.,0.],
                                [0.,0.,0.,1.,0.],
                                [0.,0.,0.,1.,0.],
                                [0.,0.,1.,0.,0.]]],
                              [[[0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.],
                                [1.,0.,0.,0.,1.],
                                [0.,1.,1.,1.,0.],
                                [0.,0.,0.,0.,0.]]],
                              [[[0.,0.,0.,0.,0.],
                                [0.,1.,1.,1.,0.],
                                [1.,0.,0.,0.,1.],
                                [0.,0.,0.,0.,0.],
                                [0.,0.,0.,0.,0.]]]
                              ],requires_grad=False))
dic_drone_curve = {}

# for i, (data, labels) in enumerate(dataloader, 0):
#     labels = labels[0].split(';')[0]
#     # print(i)
#     # print(D(data).size())
#     if labels not in dic_drone_curve.keys():
#         dic_drone_curve[labels] = []
#         dic_drone_curve[labels].append(D(data).sum().item())
#     else:dic_drone_curve[labels].append(D(data).sum().item())
#
# for key in dic_drone_curve:
#     dic_drone_curve[key] = np.asarray(dic_drone_curve[key]).sum()/(21*4736)
# dic_drone_curve = sorted(dic_drone_curve.items(), key=lambda x: x[1], reverse=True)
# for i in dic_drone_curve:
#     print((i))



for i, (data, labels) in enumerate(dronedesignloader, 0):
    labels = labels[0].split(';')[0]
    # print(i)
    # print(D(data).size())
    if labels not in dic_drone_curve.keys():
        dic_drone_curve[labels] = []
        dic_drone_curve[labels].append(D(data).sum().item())
    else:dic_drone_curve[labels].append(D(data).sum().item())

for key in dic_drone_curve:
    dic_drone_curve[key] = np.asarray(dic_drone_curve[key]).sum()/(106*4736)
dic_drone_curve = sorted(dic_drone_curve.items(), key=lambda x: x[1], reverse=True)
for i in dic_drone_curve:
    print((i))
