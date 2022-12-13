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



# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
os.makedirs("images2", exist_ok=True)
os.makedirs("/Users/lichen/Desktop/test/WGAN_generated", exist_ok=True)
# Root directory for dataset
dataroot = '/Users/lichen/Desktop/test/data_labels.csv'

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr_D = 0.00004
lr_G = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Training Rate
tr = 2

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

train_data= MyDataset(root=dataroot,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))

def classification_loss_generater(pre,real):
    listpre = []
    for i in pre:
        listpre.append(1.0 if i.item() <0.5 else 0.0)
    listpre = torch.FloatTensor(listpre)
    loss = real - listpre
    a = 0
    b = 0
    for i in loss:
        if i.item() != 0:
            a+=1
            b+=1
        else:b+=1

    return a/b

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def list_float(list):
    new_list = []
    for i in list:
        tem = []
        for j in i:
            tem.append(float(j))
        new_list.append(tem)
    return new_list

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


netG =Generator(ngpu).to(device)
netG.load_state_dict(torch.load('/Users/lichen/Desktop/test/WGANs/G_5800.pt'))

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.feature_map = []
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*8) x 4 x 4
        self.l1 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                nn.Sigmoid())
        # state size. (ndf*8) x 1 x 1
        self.poool = torch.nn.MaxPool2d(4, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    def forward(self, input):
        xx1 = self.main(input)
        self.feature_map = self.poool(xx1)
        xx2 = self.l1(xx1)
        return xx2


# Create the Discriminator
netD = Discriminator(ngpu).to(device)
netD.load_state_dict(torch.load('/Users/lichen/Desktop/test/WGANs/D_5800.pt'))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
fixed_noise = torch.randn(200, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.


img_list = []
real_fake_list = []
iters = 0
real_feature_maps=[]
fake_feature_maps=[]
for epoch in range(num_epochs):
    time_start = time.time()
    # For each batch in the dataloader
    for i, (data, labels) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        if i != None:
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            outputforloss_real = output.view(-1)
            print(classification_loss_generater(outputforloss_real,label))

        # # Calculate loss on all-real batch



        #     feature_map_real = netD.feature_map
        #     print('real')
        #     store_temp = torch.reshape(feature_map_real, (feature_map_real.size()[0], 256))
        #     temlist = list_float(store_temp.tolist())
        #     for kk in range(0,len(temlist)):
        #         temlist[kk].append(labels[kk])
        #     real_feature_maps = real_feature_maps + temlist
        #     # real_feature_maps.append()
        #
        #
        #
        #     ## Train with all-fake batch
        #     # Generate batch of latent vectors
        #     noise = torch.randn(b_size, nz, 1, 1, device=device)
        #     # Generate fake image batch with G
        #     fake = netG(noise)
        #     label.fill_(fake_label)
        #     # Classify all fake batch with D
        #     output = netD(fake.detach()).view(-1)
        #     feature_map_fake = netD.feature_map
        #     print('fake')
        #     store_temp = torch.reshape(feature_map_fake, (feature_map_fake.size()[0], 256))
        #     fake_feature_maps = fake_feature_maps + store_temp.tolist()
        #
        #
        #     # with torch.no_grad():
        #     #     fake = netG(fixed_noise).detach().cpu()
        #     fake = fake.detach().cpu()
        #     output = output.detach().cpu()
        #     # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        #     img_list+=fake
        #     real_fake_list += output
        # print(time.time() - time_start)

# D_labels = []
# for i in range(0,len(img_list)):
#     plt.figure(figsize=(15, 15))
#     plt.axis("off")
#     plt.imshow(np.transpose(vutils.make_grid(img_list[i], padding=2, normalize=True), (1, 2, 0)))
#     plt.savefig('/Users/lichen/Desktop/test/WGAN_generated/Single_Generated_%d.png' % i)
#     fake_feature_maps[i].append(i)
#     D_labels.append([i,1 if real_fake_list[i] <=0.5 else 0])
# print(real_fake_list)


# with open('/Users/lichen/Desktop/test/WGAN_generated/real_fake_list.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerows(D_labels)
#
# fake_feature_maps=list_float(fake_feature_maps)
# # real_feature_maps=list_float(real_feature_maps)
# # print(fake_feature_maps)
# with open('/Users/lichen/Desktop/test/WGAN_generated/real_feature_maps.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerows(real_feature_maps)
#
# with open('/Users/lichen/Desktop/test/WGAN_generated/fake_feature_maps.csv', 'w') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerows(fake_feature_maps)