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
from torchsummary import summary
import torchvision.models as models


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Root directory for dataset

dataroot_patch1 = 'datapath'
# dataroot_patch1 = 'rotation_original.csv'

# Number of workers for dataloader
workers = 0

num_classes = 15

# Batch size during training
batch_size = 8

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 224

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr_D = 0.00001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Training Rate
# tr = 2


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        fh = open(root, 'r')
        imgs = []
        for line in fh:
            if line != '\n':
                words = line.split(',')
                tem = int(words[1])
                imgs.append((words[0],tem))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = PIL.Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img,label,fn
    def __len__(self):
        return len(self.imgs)



patch1= MyDataset(root=dataroot_patch1,transform=transforms.Compose([
    transforms.Resize(25),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]))


dataloader1 = torch.utils.data.DataLoader(patch1, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


network = models.resnet18(pretrained=False)
network.fc = nn.Linear(512, num_classes)
network.load_state_dict(torch.load('model_path'))
network = network.eval().cuda()

network.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(network, list(range(ngpu)))

network.eval()
summary(network,(3,image_size,image_size))


acclist = []
diction = []

for i, [data, label,path] in enumerate(dataloader1, 0):
    real_cpu = data.to(device)
    output_real = network(real_cpu)
    #
    # img = real_cpu[1, :, :, ]
    # img = transforms.ToPILImage()(img.detach().cpu())
    # img.show()


    # # Calculate loss on all-real batch
    label = label.to(device)

    pred = torch.argmax(output_real,1)
    acc = (pred == label).sum()/len(label) * 100
    print(acc)
    print(pred)
    print(label)
    print(path)
    acclist.append(acc)
    for i in range(0,len(pred)):
        diction.append([int((path[i].split('/')[-1]).split('.')[0]),label[i],pred[i],output_real[i]])

# for kakak in range(0,len(acclist)):
#         print('Drone Classification Accuracy with Patch {}: {:.5f}%'.format(str(kakak+1),acclist[kakak]))



diction.sort()
for kk in diction:
    print('Image:{}; Label:{}; Prediction:{}; rate:{}'.format(kk[0],kk[1],kk[2],kk[3]))