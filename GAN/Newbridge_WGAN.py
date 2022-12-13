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
from torch.autograd import Variable

# Set random seed for reproducibility
manualSeed = 999

# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
os.makedirs("/Users/lichen/PyTorch-GAN/images_wgan/", exist_ok=True)
os.makedirs('/Users/lichen/PyTorch-GAN/WGANs/', exist_ok=True)


# Root directory for dataset
# dataroot = "/Users/lichen/Desktop/test/1model"

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
num_epochs = 2001

# Learning rate for optimizers
lr_D = 0.0004
lr_G = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Training Rate
tr = 2

#
# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=workers)
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

train_data= MyDataset(root='/Users/lichen/Desktop/test/data_labels.csv',transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# flag = torch.cuda.is_available()
# if flag:
#     print("CUDA可使用")
# else:
#     print("CUDA不可用")
#
# print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))


# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.close()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



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


######################################################################
# Now, we can instantiate the generator and apply the ``weights_init``
# function. Check out the printed model to see how the generator object is
# structured.
#

# Create the generator
netG = Generator(ngpu).to(device)


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
# print(netG)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.l1 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        x1 = self.main(input)
        x2 = self.l1(x1)
        return x2


######################################################################
# Now, as with the generator, we can create the discriminator, apply the
# ``weights_init`` function, and print the model’s structure.
#

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)

criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
optimizer_G = torch.optim.RMSprop(netG.parameters(), lr=lr_G)
optimizer_D = torch.optim.RMSprop(netD.parameters(), lr=lr_D)

img_list = []
G_losses = []
D_losses = []
iters = 0
STARTTIME = time.time()
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    time_start = time.time()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        # ############################
        # # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # ###########################
        # ## Train with all-real batch
        # netD.zero_grad()
        # # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # # Forward pass real batch through D

        output_real = netD(real_cpu)
        outputforloss_real = output_real.view(-1)
        # # Calculate loss on all-real batch
        errD_real = criterion(outputforloss_real, label)
        # # Calculate gradients for D in backward pass
        # errD_real.backward()
        # D_x = output.mean().item()
        #
        # ## Train with all-fake batch
        # # Generate batch of latent vectors
        # noise = torch.randn(b_size, nz, 1, 1, device=device)
        # # Generate fake image batch with G
        # fake = netG(noise)
        # label.fill_(fake_label)
        # # Classify all fake batch with D
        # output = netD(fake.detach()).view(-1)
        # # Calculate D's loss on the all-fake batch
        # errD_fake = criterion(output, label)
        # # Calculate the gradients for this batch
        # errD_fake.backward()
        # D_G_z1 = output.mean().item()
        # # Add the gradients from the all-real and all-fake batches
        # errD = errD_real + errD_fake
        # # Update D
        # optimizerD.step()





        optimizer_D.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        z = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate a batch of images
        fake_imgs = netG(z).detach()
        output_fake = netD(fake_imgs)
        label.fill_(fake_label)
        outputforloss_fake = output_fake.view(-1)
        errD_fake = criterion(outputforloss_fake, label)
        errD = errD_real + errD_fake
        # Adversarial loss
        loss_D = -torch.mean(output_real) + torch.mean(output_fake)

        loss_D.backward()
        optimizer_D.step()
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        if i % 2 == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = netG(z)
            # Adversarial loss
            tempG = netD(gen_imgs)
            label.fill_(real_label)
            errG = criterion(tempG.view(-1), label)
            loss_G = -torch.mean(tempG)

            loss_G.backward()
            optimizer_G.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(),errG.item()))




        # ############################
        # # (2) Update G network: maximize log(D(G(z)))
        # ###########################
        # # for cc in range(0,tr):
        # netG.zero_grad()
        # label.fill_(real_label)  # fake labels are real for generator cost
        # # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = netD(fake).view(-1)
        # # Calculate G's loss based on this output
        # errG = criterion(output, label)
        # # Calculate gradients for G
        # errG.backward()
        # D_G_z2 = output.mean().item()
        # # Update G
        # optimizerG.step()
        #
        #
        # # Output training stats
        # if i % 1 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, num_epochs, i, len(dataloader),
        #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        if epoch % 100 == 0:
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                                    (1, 2, 0)))

            # Plot the fake images from the last epoch
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            # plt.show()
            plt.savefig('/Users/lichen/PyTorch-GAN/images_wgan/Generated_%d.png' % epoch)
            torch.save(netD.state_dict(), '/Users/lichen/PyTorch-GAN/WGANs/'+'D_'+str(epoch)+'.pt')
            torch.save(netG.state_dict(), '/Users/lichen/PyTorch-GAN/WGANs/' + 'G_' + str(epoch) + '.pt')
            plt.close()
        iters += 1
    print(time.time() - time_start)
######################################################################
# Results
# -------
#
# Finally, lets check out how we did. Here, we will look at three
# different results. First, we will see how D and G’s losses changed
# during training. Second, we will visualize G’s output on the fixed_noise
# batch for every epoch. And third, we will look at a batch of real data
# next to a batch of fake data from G.
#
# **Loss versus training iteration**
#
# Below is a plot of D & G’s losses versus training iterations.
#

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('/Users/lichen/PyTorch-GAN/images_wgan/Loss.jpg')

######################################################################
# **Visualization of G’s progression**
#
# Remember how we saved the generator’s output on the fixed_noise batch
# after every epoch of training. Now, we can visualize the training
# progression of G with an animation. Press the play button to start the
# animation.
#

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
# HTML(ani.to_jshtml())


real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
# plt.show()
plt.savefig('/Users/lichen/PyTorch-GAN/images_wgan/Generated.jpg')
ENDTIME = time.time()
print(ENDTIME-STARTTIME)