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
# Root directory for dataset
dataroot = '/Users/lichen/Desktop/test/data_labels.csv'
dataroot_R = '/Users/lichen/Desktop/test/additional_data_labels_R.csv'
dataroot_GT = '/Users/lichen/Desktop/test/additional_data_labels_GAN_TDA.csv'
dataroot_test = '/Users/lichen/Desktop/test/model_test_set.csv'
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
num_epochs = 200

# Learning rate for optimizers
lr_D = 0.00001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Training Rate
tr = 2

class MyDataset2(torch.utils.data.Dataset):
    def __init__(self,root1, root2, transform=None, target_transform=None):
        super(MyDataset2,self).__init__()
        fh1 = open(root1, 'r')
        fh2 = open(root2, 'r')
        imgs = []
        for line in fh1:
            words = line.split(',')
            imgs.append((words[0],words[1]))
        for line in fh2:
            words = line.split(',')
            imgs.append((words[0],words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.root1 = root1
        self.root2 = root2

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = PIL.Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class MyDataset1(torch.utils.data.Dataset):
    def __init__(self,root1, transform=None, target_transform=None):
        super(MyDataset1,self).__init__()
        fh1 = open(root1, 'r')
        imgs = []
        for line in fh1:
            words = line.split(',')
            imgs.append((words[0],words[1]))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = PIL.Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)



train_data_R= MyDataset2(root1=dataroot,root2=dataroot_R,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

train_data_GT= MyDataset2(root1=dataroot,root2=dataroot_GT,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

test= MyDataset1(root1=dataroot_test,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader_R = torch.utils.data.DataLoader(train_data_R, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
dataloader_GT = torch.utils.data.DataLoader(train_data_GT, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
dataloader_test = torch.utils.data.DataLoader(test,batch_size=100,
                                              shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



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
        # # state size. (ndf*8) x 1 x 1
        # self.poool = torch.nn.MaxPool2d(4, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    def forward(self, input):
        xx1 = self.main(input)
        xx2 = self.l1(xx1)
        return xx2


# Create the Discriminator for Randomly collection of new data
netD_R = Discriminator(ngpu).to(device)
netD_R.load_state_dict(torch.load('/Users/lichen/Desktop/test/WGANs/D_5800.pt'))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_R = nn.DataParallel(netD_R, list(range(ngpu)))

fixed_noise = torch.randn(200, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD_R.parameters(), lr=lr_D, betas=(beta1, 0.999))



def miss_calculater(a,b):
    a = a.squeeze()
    a = torch.as_tensor((a - 0.5) > 0, dtype=torch.int32)
    result = a-b
    abssum = torch.sum(torch.abs(a-b))
    return abssum/len(result)


loss_R_train = []
loss_R_test = []
loss_GT_train = []
loss_GT_test = []
GT_b_loss = []
R_b_loss = []


for epoch in range(0,num_epochs):
    time_start = time.time()
    # For each batch in the dataloader
    for i, (data, labels) in enumerate(dataloader_R, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        if i != None:
            netD_R.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output_real = netD_R(real_cpu)
            outputforloss_real = output_real.view(-1)
            loss_real = criterion(outputforloss_real,label)
            loss_real.backward()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise).detach()
            label.fill_(fake_label)
            # Classify all fake batch with D
            output_fake = netD_R(fake)
            outputforloss_fake = output_fake.view(-1)
            loss_fake = criterion(outputforloss_fake,label)
            loss_fake.backward()
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            # loss_D = -torch.mean(output_real) + torch.mean(output_fake)

            optimizerD.step()

            print(
                "[Randomly] [Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                % (epoch, num_epochs, i, len(dataloader_R), (loss_fake+loss_real).item())
            )


    loss_R_train.append((loss_fake+loss_real).item())

    loss_1 = 0 #avg for real
    loss_2 = 0 #avg for fake
    for i, (data, labels) in enumerate(dataloader_test, 0):

        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output_real = netD_R(real_cpu)
        outputforloss_real = output_real.view(-1)
        loss_real_c = criterion(outputforloss_real,label)
        loss_real = miss_calculater(outputforloss_real,label)
        loss_1 += loss_real

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise).detach()
        label.fill_(fake_label)
        output_fake = netD_R(fake)
        outputforloss_fake = output_fake.view(-1)
        loss_fake_c = criterion(outputforloss_fake,label)
        loss_fake = miss_calculater(outputforloss_fake,label)
        loss_2 += loss_fake
    print(
        "[Randomly] [Epoch %d/%d] [D loss: %f] [classification precision: %f | %f] [time: %f]"
        % (epoch, num_epochs, (loss_fake + loss_real).item(), loss_real, loss_fake,time.time() - time_start)
    )
    loss_R_test.append((loss_fake_c + loss_real_c).item())
    R_b_loss.append([loss_1.item()/len(dataloader_test),loss_2.item()/len(dataloader_test)])



# Create the Discriminator for GAN-TDA of new data
netD_GT = Discriminator(ngpu).to(device)
netD_GT.load_state_dict(torch.load('/Users/lichen/Desktop/test/WGANs/D_5800.pt'))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_GT = nn.DataParallel(netD_GT, list(range(ngpu)))

fixed_noise = torch.randn(200, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

criterion = nn.BCELoss()
optimizer_D = optim.Adam(netD_GT.parameters(), lr=lr_D, betas=(beta1, 0.999))

for epoch in range(0,num_epochs):
    time_start = time.time()
    # For each batch in the dataloader
    for i, (data, labels) in enumerate(dataloader_GT, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        if i != None:
            netD_GT.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output_real = netD_GT(real_cpu)
            outputforloss_real = output_real.view(-1)
            loss_real = criterion(outputforloss_real,label)
            loss_real.backward()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise).detach()
            label.fill_(fake_label)
            # Classify all fake batch with D
            output_fake = netD_GT(fake)
            outputforloss_fake = output_fake.view(-1)
            loss_fake = criterion(outputforloss_fake,label)
            loss_fake.backward()
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            # loss_D = -torch.mean(output_real) + torch.mean(output_fake)

            optimizer_D.step()

            print(
                "[GAN-TDA] [Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                % (epoch, num_epochs, i, len(dataloader_GT), (loss_fake+loss_real).item())
            )


    loss_GT_train.append((loss_fake+loss_real).item())

    loss_1 = 0 #avg for real
    loss_2 = 0 #avg for fake
    for i, (data, labels) in enumerate(dataloader_test, 0):

        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output_real = netD_GT(real_cpu)
        outputforloss_real = output_real.view(-1)
        loss_real_c = criterion(outputforloss_real,label)
        loss_real = miss_calculater(outputforloss_real,label)
        loss_1 += loss_real

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise).detach()
        label.fill_(fake_label)
        output_fake = netD_GT(fake)
        outputforloss_fake = output_fake.view(-1)
        loss_fake_c = criterion(outputforloss_fake,label)
        loss_fake = miss_calculater(outputforloss_fake,label)
        loss_2 += loss_fake
    print(
        "[GAN-TDA] [Epoch %d/%d] [D loss: %f] [classification precision: %f | %f] [time: %f]"
        % (epoch, num_epochs, (loss_fake + loss_real).item(), loss_real, loss_fake,time.time() - time_start)
    )
    loss_GT_test.append((loss_fake_c + loss_real_c).item())
    GT_b_loss.append([loss_1.item()/len(dataloader_test),loss_2.item()/len(dataloader_test)])
torch.save(netD_GT.state_dict(), '/Users/lichen/Desktop/test/validation_performance1/' + 'D_GT' + str(epoch) + '.pt')
torch.save(netD_R.state_dict(), '/Users/lichen/Desktop/test/validation_performance1/' + 'D_R' + str(epoch) + '.pt')
with open('/Users/lichen/Desktop/test/validation_performance1/model_loss_R_train.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(loss_R_train)
with open('/Users/lichen/Desktop/test/validation_performance1/model_loss_R_test.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(loss_R_test)
with open('/Users/lichen/Desktop/test/validation_performance1/model_loss_GT_train.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(loss_GT_train)
with open('/Users/lichen/Desktop/test/validation_performance1/model_loss_GT_test.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(loss_GT_test)
with open('/Users/lichen/Desktop/test/validation_performance1/model_GT_b_loss.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(GT_b_loss)
with open('/Users/lichen/Desktop/test/validation_performance1/model_R_b_loss.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(R_b_loss)



