"""
Training code for Adversarial patch training


"""

import PIL
import torch

import load_data
from tqdm import tqdm

from passanger_load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import albumentations as A

import passanger_patch_config as patch_config
import sys
import time

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.patch_shape = self.patch_shaper('C:/Users/s324652/OneDrive - Cranfield University/Desktop/patch_shape/star.png')

        self.transformerset = transforms.RandomApply(
            transforms=[transforms.RandomCrop(size=(360, 360)),
                        transforms.RandomRotation(degrees=(-10, 10)),
                        # transforms.Resize(size=torch.randint(int(0.8 * 416), 416, (1, 1)).item())
                        ], p=0.7)

        self.writer = self.init_tensorboard(mode)


    def init_tensorboard(self, name=None):
        # pass
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S-passenger")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 500
        max_lab = 16

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        # adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("C:/Users/s324652/OneDrive - Cranfield University/Desktop/ren.jpg")
        adv_patch_cpu = self.read_image("C:/Users/s324652/OneDrive - Cranfield University/Desktop/patch_shape/patchnew_499.jpg")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                adv_patch = adv_patch_cpu.cuda()

                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, self.patch_shape,do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                # p_img_batch = transforms.Resize(size=torch.randint(int(0.5 * 416), 416, (1, 1)).item())(p_img_batch)
                p_img_batch = self.transformerset(p_img_batch)
                p_img_batch = transforms.Resize(size=416)(p_img_batch)


                q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                # img = q_img_batch[0, :, :, ]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()

                output = self.darknet_model(q_img_batch)
                max_prob = self.prob_extractor(output)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)


                nps_loss = nps*0.01
                tv_loss = tv*0.5
                det_loss = torch.mean(max_prob)
                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                # loss = det_loss

                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                bt1 = time.time()
                if i_batch%5 == 0:
                    iteration = self.epoch_length * epoch + i_batch

                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/epoch', epoch, iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    self.writer.add_image('patch', adv_patch_cpu, iteration)
                if i_batch + 1 >= len(train_loader):
                    print('\n')
                else:
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    torch.cuda.empty_cache()
                bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu * (self.patch_shape.cpu()))

                # plt.imshow(im)
                # # plt.savefig('saved_patches/patchnew_' + str(epoch) + '.jpg')
                # plt.show()
                im.save('ps_saved_patches/patchnew_'+str(epoch)+'.jpg')

                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

            # -----------------------------------varification---------------------------------------
            ep_det_rate = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_det_loss = 0


            validation_loader = torch.utils.data.DataLoader(
                InriaDataset(self.config.img_dir_v, self.config.lab_dir_v, max_lab, img_size,
                             shuffle=True),
                batch_size=batch_size,
                shuffle=True,
                num_workers=10)
            self.epoch_length_v = len(validation_loader)
            print(f'One epoch is {len(validation_loader)}')


            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(validation_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length_v):
                with torch.no_grad():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    cls_label = lab_batch.clone().detach()[:, 0, 0].long()

                    adv_patch = adv_patch_cpu.cuda()

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, 416, self.patch_shape,do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    # img = p_img_batch[1, :, :, ]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    output = self.darknet_model(q_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(max_prob)
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    # loss = det_loss

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    if i_batch + 1 >= len(validation_loader):
                        iteration = self.epoch_length_v * epoch + i_batch

                        self.writer.add_scalar('v_total_loss', ep_loss / len(validation_loader), iteration)
                        self.writer.add_scalar('v_loss/det_loss', ep_det_loss / len(validation_loader), iteration)


                    if i_batch + 1 >= len(validation_loader):
                        print('\n')
                        print('val cls_accuracy:', ep_det_loss / len(validation_loader))
                    else:
                        del adv_batch_t, p_img_batch, nps_loss, tv_loss, loss, q_img_batch
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            img = q_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('ps_patched_figures/figure_v_' + str(epoch) + '.jpg')

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu

    def patch_shaper(self, add):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """

        patch_img = Image.open(add).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)

        tf = transforms.ToTensor()
        patch_shape = tf(patch_img).cuda()

        patch_shape = torch.cuda.FloatTensor(patch_shape.size()).fill_(1) - patch_shape

        return patch_shape



def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer('paper_obj')
    trainer.train()

if __name__ == '__main__':
    main()


