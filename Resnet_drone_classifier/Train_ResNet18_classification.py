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
import PIL.Image
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

# Set random seed for reproducibility
manualSeed = 999

# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


store_head = 'C:/Users/s324652/Desktop/yolov2/DroneDetector/ResNet18/'
# Root directory for dataset
# dataroot = "/Users/lichen/Desktop/test/1model"

# Number of workers for dataloader
workers = 0

feature_extract = False

# Batch size during training
batch_size = 64


num_classes = 14

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 224

# Number of channels in the training images. For color images this is 3
nc = 3


# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr_D = 0.0001


# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




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
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data= MyDataset(root='model_images_14.csv',transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.ColorJitter(brightness = 0.5, contrast=0.5,hue = 0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
]))

test_data= MyDataset(root='org_validation.csv',transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.ColorJitter(brightness = 0.5, contrast=0.5,hue = 0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
]))


dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


flag = torch.cuda.is_available()

if flag:
    print("CUDA available")
else:
    print("CUDA unavailable")

print("Driver：",device)
print("GPU： ",torch.cuda.get_device_name(0))

network = models.resnet18(pretrained=True)
set_parameter_requires_grad(network, feature_extract)
network.fc = nn.Linear(512,num_classes)

network.to(device)



criterion = nn.CrossEntropyLoss()
optimizer_D = torch.optim.Adam(network.parameters(), lr=lr_D, betas = (0.9,0.999))


iters = 0
STARTTIME = time.time()
print("Starting Training Loop...")
batches = len(dataloader)
# For each epoch



loss_train = []
loss_validation = []


for epoch in range(num_epochs):

    # For each batch in the dataloader
    network.train()
    for i, [data, label] in enumerate(dataloader, 0):

        real_cpu = data.to(device)

        output_real = network(real_cpu)
        # # Calculate loss on all-real batch
        label = label.to(device)

        errD_real = criterion(output_real, label)

        pred = torch.argmax(output_real,1)
        acc = (pred == label).sum()/len(label) * 100
        optimizer_D.zero_grad()
        errD_real.backward()
        optimizer_D.step()
        print('Training: Epoch: {}/{}, Batch: {}/{}, Training Loss: {:.6f}, Classification Accuracy: {:.5f}%'.format(epoch,num_epochs,i,batches,errD_real.item(),acc))

        loss_train.append(errD_real.cpu().detach().numpy())

    network.eval()

    with torch.no_grad():
        loss_test = 0
        for i, [data, label] in enumerate(dataloader_test, 0):
            real_cpu = data.to(device)
            output_real = network(real_cpu)
            label = label.to(device)
            errD_real = criterion(output_real, label)
            loss_test+=errD_real
            pred = torch.argmax(output_real, 1)
            acc = (pred == label).sum() / len(label) * 100
        print('Testing: Validation Loss: {:.6f}, Classification Accuracy: {:.5f}%'.format((loss_test/len(dataloader_test)).item(),acc))
        loss_validation.append((loss_test/len(dataloader_test)).cpu().detach().numpy())


    print(pred)
    print(label)
    torch.save(network.state_dict(), store_head+'D_'+str(epoch)+'.pt')

# plt.figure()
# plt.title('loss')
# plt.yscale('log')
# # plt.plot(loss_validation,label = 'velidation')
# plt.plot(loss_train,label = 'train')
# plt.xlabel('Iterations')
# plt.ylabel('CE loss')
# plt.legend()
# plt.savefig(store_head+'training_loss.jpg')
#
#
#
# plt.figure()
# plt.title('loss')
# plt.yscale('log')
# plt.plot(loss_validation,label = 'validation')
# # plt.plot(loss_train,label = 'train')
# plt.xlabel('epochs')
# plt.ylabel('CE loss')
# plt.legend()
# plt.savefig(store_head+'validation_loss.jpg')