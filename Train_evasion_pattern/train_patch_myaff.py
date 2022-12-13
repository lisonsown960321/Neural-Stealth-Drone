"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data_my_version import *
# from passanger_load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
import torchvision.models as models
import torch.nn as nn
import torch
import darknet
from albumentations import Blur


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.num_classes = 15
        self.darknet_model = darknet.Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)

        # self.darknet_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained = True).eval()
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        # self.darknet_model.classes = [0]
        self.darknet_model.max_det = 5
        # self.darknet_model_height =320
        print('reload YOLO success')


        self.Resnet_classifier = models.resnet18(pretrained=False)
        self.Resnet_classifier.fc = nn.Linear(512, self.num_classes)
        self.Resnet_classifier.load_state_dict(torch.load('C:/Users/s324652/Desktop/yolov2/DroneDetector/ResNet18/D_8.pt'))
        self.Resnet_classifier = self.Resnet_classifier.eval().cuda()
        print('reload Resnet success')

        self.transform = transforms.Resize(224)


        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer_new().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.criterion = nn.CrossEntropyLoss()


        self.writer = self.init_tensorboard(mode)


    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S-1")
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
        n_epochs = 300
        max_lab = 15

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        validation_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir_v, self.config.lab_dir_v, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        self.epoch_length_v = len(validation_loader)

        self.epoch_length = len(train_loader)


        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        # optimizer = optim.RMSprop([adv_patch_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()


        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_cls_loss = 0
            ep_cls_rate = 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                cls_label = lab_batch.clone().detach()[:, 0, 0].long()

                # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                adv_patch = adv_patch_cpu.cuda()

                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                p_img_batch = transforms.Resize(size=torch.randint(int(0.5 * img_size), img_size, (1, 1)).item())(p_img_batch)
                p_img_batch = transforms.Resize(size=416)(p_img_batch)

                #
                #
                # img = adv_patch
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()

                r_img_batch = self.transform(p_img_batch)
                q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                #


                output = self.darknet_model(q_img_batch)

                classification_output = self.Resnet_classifier(r_img_batch)

                cls = self.criterion(classification_output, cls_label)

                max_prob = self.prob_extractor(output)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                cls_discount = 1
                det_discount = 1

                cls_loss = cls*cls_discount
                nps_loss = nps* 0.1
                tv_loss = tv * 1.5
                det_loss = torch.mean(max_prob)* det_discount

                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) - cls_loss
                # loss = -cls_loss

                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_cls_loss += cls_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0.0001, 0.99999)       #keep patch in image range

                bt1 = time.time()
                if i_batch%5 == 0:

                    iteration = self.epoch_length * epoch + i_batch

                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/cls_loss', cls.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy()/det_discount, iteration)
                    self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    self.writer.add_image('patch', adv_patch_cpu, iteration)

                if i_batch + 1 >= len(train_loader):
                    print('\n')
                    pred = torch.argmax(classification_output, 1)
                    cls_rate = (pred == cls_label).sum() / len(cls_label) * 100
                    print('cls_accuracy:', cls_rate)
                else:
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,r_img_batch,q_img_batch
                    torch.cuda.empty_cache()
                bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_cls_loss = ep_cls_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #
            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            img = r_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('patched_figures_both/figure_' + str(epoch) + '.jpg')

            scheduler.step(ep_loss)
            if True:
                print('  Training: ', 'both'),
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  CLS LOSS: ', ep_cls_loss/cls_discount)
                print('  DET LOSS: ', ep_det_loss/det_discount)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                im.save('saved_patches_both/patchnew_'+str(epoch)+'.jpg')

                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,r_img_batch,q_img_batch
                torch.cuda.empty_cache()
            et0 = time.time()

            #  validation
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_cls_loss = 0
            ep_cls_rate = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(validation_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length_v):
                with torch.no_grad():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    cls_label = lab_batch.clone().detach()[:, 0, 0].long()

                    adv_patch = adv_patch_cpu.cuda()

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    p_img_batch = transforms.Resize(size=torch.randint(int(0.5 * img_size), img_size, (1, 1)).item())(
                        p_img_batch)
                    p_img_batch = transforms.Resize(size=416)(p_img_batch)


                    r_img_batch = self.transform(p_img_batch)
                    q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    output = self.darknet_model(q_img_batch)

                    classification_output = self.Resnet_classifier(r_img_batch)

                    pred = torch.argmax(classification_output, 1)
                    cls_rate = (pred == cls_label).sum() / len(cls_label) * 100

                    cls = self.criterion(classification_output, cls_label)

                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    cls_loss = cls  # 0.5
                    nps_loss = nps
                    tv_loss = tv
                    det_loss = torch.mean(max_prob)

                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) - cls_loss


                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_cls_loss += cls_loss.detach().cpu().numpy()
                    ep_loss += loss.detach().cpu().numpy()
                    ep_cls_rate += cls_rate.detach().cpu().numpy()

                    if i_batch%1 == 0:
                        iteration = self.epoch_length_v * epoch + i_batch

                        self.writer.add_scalar('v_total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/cls_loss', cls.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/cls_rate', cls_rate.detach().cpu().numpy(), iteration)


                    if i_batch + 1 >= len(validation_loader):
                        print('\n')

                        print('val cls_accuracy:', ep_cls_rate/len(validation_loader))
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,r_img_batch,q_img_batch
                        torch.cuda.empty_cache()
                    bt0 = time.time()

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




'''-------------------------------------------------------------------------------------------------'''
class PatchTrainer_cls(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.num_classes = 15

        self.Resnet_classifier = models.resnet18(pretrained=False)
        self.Resnet_classifier.fc = nn.Linear(512, self.num_classes)
        self.Resnet_classifier.load_state_dict(torch.load('C:/Users/s324652/Desktop/yolov2/DroneDetector/ResNet18/D_8.pt'))
        self.Resnet_classifier = self.Resnet_classifier.eval().cuda()
        print('reload Resnet success')

        self.transform = transforms.Resize(224)


        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer_new().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S-2")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        batch_size = self.config.batch_size
        n_epochs = 300
        max_lab = 15

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, 416,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        validation_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir_v, self.config.lab_dir_v, max_lab, 416,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length_v = len(validation_loader)

        self.epoch_length = len(train_loader)


        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        # optimizer = optim.RMSprop([adv_patch_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()

        for epoch in range(n_epochs):

            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_cls_loss = 0
            ep_cls_rate = 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                cls_label = lab_batch.clone().detach()[:, 0, 0].long()

                # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                adv_patch = adv_patch_cpu.cuda()

                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, 416, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                p_img_batch = transforms.Resize(size=torch.randint(int(0.1 *416), 416, (1, 1)).item())(p_img_batch)
                p_img_batch = transforms.Resize(size=416)(p_img_batch)


                r_img_batch = self.transform(p_img_batch)

                classification_output = self.Resnet_classifier(r_img_batch)

                cls = self.criterion(classification_output, cls_label)

                pred = torch.argmax(classification_output, 1)
                cls_rate = (pred == cls_label).sum() / len(cls_label) * 100

                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                cls_discount = 1


                cls_loss = cls*cls_discount
                nps_loss = nps* 0.1
                tv_loss = tv * 1.5


                loss = nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) - cls_loss
                # loss = -cls_loss

                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_cls_loss += cls_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()
                ep_cls_rate += cls_rate.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0.0001, 0.99999)       #keep patch in image range

                bt1 = time.time()
                if i_batch%5 == 0:

                    iteration = self.epoch_length * epoch + i_batch

                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/cls_loss', cls.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    self.writer.add_image('patch', adv_patch_cpu, iteration)

                if i_batch + 1 >= len(train_loader):
                    print('\n')

                else:
                    del adv_batch_t,p_img_batch, nps_loss, tv_loss, loss,r_img_batch
                    torch.cuda.empty_cache()
                bt0 = time.time()
            et1 = time.time()

            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_cls_loss = ep_cls_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            ep_cls_rate = ep_cls_rate/ len(train_loader)

            img = r_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('patched_figures_cls/figure_'+str(epoch)+'.jpg')

            scheduler.step(ep_loss)
            if True:
                print('  Training: ', 'cls'),
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  CLS LOSS: ', ep_cls_loss/cls_discount)
                print('  CLS Rate: ', ep_cls_rate)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                im.save('saved_patches_cls/patchnew_'+str(epoch)+'.jpg')

                del adv_batch_t, p_img_batch, nps_loss, tv_loss, loss,r_img_batch
                torch.cuda.empty_cache()
            et0 = time.time()

            #  validation
            ep_cls_rate = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_cls_loss = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(validation_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length_v):
                with torch.no_grad():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    cls_label = lab_batch.clone().detach()[:, 0, 0].long()

                    adv_patch = adv_patch_cpu.cuda()

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, 416, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = transforms.Resize(size=torch.randint(int(0.1 * 416), 416, (1, 1)).item())(
                        p_img_batch)
                    p_img_batch = transforms.Resize(size=416)(p_img_batch)


                    r_img_batch = self.transform(p_img_batch)

                    classification_output = self.Resnet_classifier(r_img_batch)

                    pred = torch.argmax(classification_output, 1)
                    cls_rate = (pred == cls_label).sum() / len(cls_label) * 100

                    cls = self.criterion(classification_output, cls_label)

                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    cls_loss = cls  # 0.5
                    nps_loss = nps
                    tv_loss = tv

                    loss = nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) - cls_loss


                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_cls_loss += cls_loss.detach().cpu().numpy()
                    ep_loss += loss.detach().cpu().numpy()
                    ep_cls_rate += cls_rate.detach().cpu().numpy()

                    if i_batch + 1 >= len(validation_loader):
                        iteration = self.epoch_length_v * epoch + i_batch

                        self.writer.add_scalar('v_total_loss', ep_loss/len(validation_loader), iteration)
                        self.writer.add_scalar('v_loss/cls_loss', ep_cls_loss/len(validation_loader), iteration)
                        self.writer.add_scalar('v_loss/cls_rate', ep_cls_rate/len(validation_loader), iteration)

                    if i_batch + 1 >= len(validation_loader):
                        print('\n')

                        print('val cls_accuracy:', ep_cls_rate/len(validation_loader))
                    else:
                        del adv_batch_t, p_img_batch, nps_loss, tv_loss, loss,r_img_batch
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            img = r_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('patched_figures_cls/figure_v_' + str(epoch) + '.jpg')

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




'''-------------------------------------------------------------------------------------------------------'''
class PatchTrainer_obj(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.num_classes = 15
        self.darknet_model = darknet.Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)

        # self.darknet_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained = True).eval()
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        # self.darknet_model.classes = [0]
        self.darknet_model.max_det = 5
        # self.darknet_model_height =320
        print('reload YOLO success')



        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer_new().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)


    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S-3")
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
        n_epochs = 100
        max_lab = 15

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

        validation_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir_v, self.config.lab_dir_v, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)


        self.epoch_length_v = len(validation_loader)

        self.epoch_length = len(train_loader)


        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        # optimizer = optim.RMSprop([adv_patch_cpu], lr=self.config.start_learning_rate)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()


        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()

                adv_patch = adv_patch_cpu.cuda()

                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                p_img_batch = transforms.Resize(size=torch.randint(int(0.5 * 416), 416, (1, 1)).item())(p_img_batch)
                p_img_batch = transforms.Resize(size=416)(p_img_batch)



                q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))


                output = self.darknet_model(q_img_batch)


                max_prob = self.prob_extractor(output)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                det_discount = 1

                nps_loss = nps* 0.1
                tv_loss = tv * 1.5
                det_loss = torch.mean(max_prob)* det_discount

                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                ep_det_loss += det_loss.detach().cpu().numpy()
                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0.0001, 0.99999)       #keep patch in image range

                bt1 = time.time()
                if i_batch%5 == 0:

                    iteration = self.epoch_length * epoch + i_batch

                    self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy()/det_discount, iteration)
                    self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    self.writer.add_image('patch', adv_patch_cpu, iteration)

                if i_batch + 1 >= len(train_loader):
                    print('\n')
                else:
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,q_img_batch
                    torch.cuda.empty_cache()
                bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            scheduler.step(ep_loss)

            img = q_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('patched_figures_obj/figure_' + str(epoch) + '.jpg')
            if True:
                print('  Training: ', 'obj'),
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss/det_discount)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                im.save('saved_patches_obj/patchnew_'+str(epoch)+'.jpg')

                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,q_img_batch
                torch.cuda.empty_cache()
            et0 = time.time()

            #  validation
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(validation_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length_v):
                with torch.no_grad():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()


                    adv_patch = adv_patch_cpu.cuda()

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = transforms.Resize(size=torch.randint(int(0.5 * 416), 416, (1, 1)).item())(
                        p_img_batch)


                    q_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    output = self.darknet_model(q_img_batch)


                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps
                    tv_loss = tv
                    det_loss = torch.mean(max_prob)

                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()

                    ep_loss += loss.detach().cpu().numpy()

                    if i_batch + 1 >= len(validation_loader):
                        iteration = self.epoch_length_v * epoch + i_batch

                        self.writer.add_scalar('v_total_loss', ep_loss, iteration)
                        self.writer.add_scalar('v_loss/det_loss', ep_det_loss, iteration)
                        self.writer.add_scalar('v_loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('v_loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)


                    if i_batch + 1 >= len(validation_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss,q_img_batch
                        torch.cuda.empty_cache()
                    bt0 = time.time()

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

def main():


    trainer = PatchTrainer('paper_obj')
    trainer.train()

    # trainer = PatchTrainer_cls('paper_obj')
    # trainer.train()

    # trainer = PatchTrainer_obj('paper_obj_obj')
    # trainer.train()

if __name__ == '__main__':
    main()


