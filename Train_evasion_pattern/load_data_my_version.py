import fnmatch
import math
import os
import sys
import time
from operator import itemgetter
import random
import pandas

import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image

from darknet import Darknet
from albumentations import Blur


from median_pool import MedianPool2d

import cv2


# im = Image.open('data/horse.jpg').convert('RGB')



class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)


        assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]1805
        # output_objectness = output[:, 4, :]  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf

class MaxProbExtractor_all(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor_all, self).__init__()

        self.num_cls = 80
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)


        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]1805
        # output_objectness = output[:, 4, :]  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)

        max_confs = torch.max(output, dim=1)[0]
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, :, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = max_confs * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, max_confs)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf


class MaxProb_detExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProb_detExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)

        assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        # output_objectness = output[:, 4, :]  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]


        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
        ones = torch.ones_like(max_conf)
        zeros = torch.zeros_like(max_conf)
        det = torch.where(max_conf>0.6,ones,zeros)

        lens = det.size(0)
        det = torch.sum(det,0)/lens



        return max_conf,det


class MaxProbExtractor_v5(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor_v5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # if YOLOoutput.dim() == 3:
        #     YOLOoutput = YOLOoutput.unsqueeze(0)
        # batch = YOLOoutput.size(0)

        output = YOLOoutput
        output = output.transpose(1, 2).contiguous()

        # output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output_objectness = output[:, 4, :]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # # we only care for probabilities of the class of interest (person)
        # confs_for_class = normal_confs[:, :, :]
        confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        # confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        ones = torch.ones_like(max_conf)
        zeros = torch.zeros_like(max_conf)
        det = torch.where(max_conf > 0.4, ones, zeros)

        lens = det.size(0)
        det = torch.sum(det, 0) / lens

        return max_conf,det

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """aff to simulate perspective
    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1
        self.min_brightness = -0.2
        self.max_brightness = 0
        self.noise_factor = 0.18
        self.minangle = -30 / 180 * math.pi
        self.maxangle = 30 / 180 * math.pi

        self.shminangle = -25 / 180 * math.pi
        self.shmaxangle = 25 / 180 * math.pi


        self.minscaleH = 1
        self.maxscaleH = 2

        self.minscaleW = 1
        self.maxscaleW = 1.5

        self.transformerset = torch.nn.Sequential(
            # transforms.ColorJitter(contrast=0.2, saturation=(0.6,1)),
            transforms.RandomAdjustSharpness(sharpness_factor=0,p=1),
        )


        self.medianpooler = MedianPool2d(7,same=True)

        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''


    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)


        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # Contrast, brightness and noise transforms

        # Apply gaussian blur (abandon)
        size0 = adv_batch.size(0)
        size1 = adv_batch.size(1)
        size3 = adv_batch.size(3)
        size4 = adv_batch.size(4)
        adv_batch = adv_batch.view(-1, 3, size3, size4)

        adv_batch = self.transformerset(adv_batch)
        adv_batch = adv_batch.view(size0, size1, 3, size3, size4)

        adv_batch = torch.clamp(adv_batch, 0.03, 0.95)

        
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()


        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()


        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor


        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise


        # print(lab_batch)
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids[:,0,0] = 0
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        # print(cls_mask)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad+0.5), int(pad), int(pad+0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)


        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            shearangle_x = torch.cuda.FloatTensor(anglesize).uniform_(self.shminangle, self.shmaxangle)
            shearangle_y = torch.cuda.FloatTensor(anglesize).uniform_(self.shminangle, self.shmaxangle)
        else: 
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if(rand_loc):
            off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
            target_x = target_x + off_x
            off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05

        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        hightscaleindex = torch.cuda.FloatTensor(anglesize).uniform_(self.minscaleH, self.maxscaleH)
        wscaleindex = torch.cuda.FloatTensor(anglesize).uniform_(self.minscaleW, self.maxscaleW)



        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])


        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        sheartanhx = torch.tanh(shearangle_x)
        sheartanhy = torch.tanh(shearangle_y)

        # Theta4 = hight matrix
        theta4 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta4[:, 0, 0] = wscaleindex
        theta4[:, 0, 1] = 0
        theta4[:, 0, 2] = 0
        theta4[:, 1, 0] = 0
        theta4[:, 1, 1] = hightscaleindex
        theta4[:, 1, 2] = 0
        #
        grid4 = F.affine_grid(theta4, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid4, padding_mode='border', align_corners=True)
        msk_batch_t = F.grid_sample(msk_batch, grid4, padding_mode='border',align_corners=True)

        theta3 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta3[:, 0, 0] = 1
        theta3[:, 0, 1] = sheartanhx
        theta3[:, 0, 2] = (-target_x + 0.5) * 2
        theta3[:, 1, 0] = sheartanhy
        theta3[:, 1, 1] = 1
        theta3[:, 1, 2] = (-target_y + 0.5) * 2

        grid3 = F.affine_grid(theta3, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid3, padding_mode='border',align_corners=True)
        msk_batch_t = F.grid_sample(msk_batch_t, grid3, padding_mode='border',align_corners=True)

        # Theta = rotation,rescale matrix

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale


        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid,padding_mode='border',align_corners = True)
        msk_batch_t = F.grid_sample(msk_batch_t, grid,padding_mode='border',align_corners = True)
        ''''''


        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])


        adv_batch_t = torch.clamp(adv_batch_t,  0.03, 0.95)


        return adv_batch_t * msk_batch_t


class PatchTransformer_new(nn.Module):
    #    random perspective

    def __init__(self):
        super(PatchTransformer_new, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1
        self.min_brightness = -0.2
        self.max_brightness = 0
        self.noise_factor = 0.10
        self.minangle = -45 / 180 * math.pi
        self.maxangle = 45 / 180 * math.pi
        self.pooling_rate = random.randint(1, 3)
        self.medianpooler = MedianPool2d(self.pooling_rate, same=True)


        self.minscaleW = 1
        self.maxscaleW = 1.5

        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids[:, 0, 0] = 0
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)


        d1 = adv_batch.size(0)
        d2 = adv_batch.size(1)
        d3 = adv_batch.size(2)
        d4 = adv_batch.size(3)
        d5 = adv_batch.size(4)


        adv_batch = adv_batch.view(-1, d3, d4, d5)
        msk_batch = msk_batch.view(-1, d3, d4, d5)


        for_pres = torch.cat([adv_batch, msk_batch], dim=1)

        perspective_transformer = transforms.RandomPerspective(distortion_scale=0.5, p=0.8)

        for_pres = perspective_transformer(for_pres)

        adv_batch, msk_batch = for_pres.split([3, 3], dim=1)


        adv_batch = adv_batch.view(d1, d2, d3, d4, d5)
        msk_batch = msk_batch.view(d1, d2, d3, d4, d5)



        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))

        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)

        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        wscaleindex = torch.cuda.FloatTensor(anglesize).uniform_(self.minscaleW, self.maxscaleW)

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)


        # Theta4 = hight matrix
        theta4 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta4[:, 0, 0] = wscaleindex
        theta4[:, 0, 1] = 0
        theta4[:, 0, 2] = 0
        theta4[:, 1, 0] = 0
        theta4[:, 1, 1] = wscaleindex
        theta4[:, 1, 2] = 0
        #
        grid4 = F.affine_grid(theta4, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid4, padding_mode='border', align_corners=True)
        msk_batch_t = F.grid_sample(msk_batch, grid4, padding_mode='border', align_corners=True)


        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid, padding_mode='border', align_corners=True)
        msk_batch_t = F.grid_sample(msk_batch_t, grid, padding_mode='border', align_corners=True)
        ''''''

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.00001, 0.99999)

        return adv_batch_t * msk_batch_t





class PatchTransformer_nonaff(nn.Module):
    """ org method - nonaff, non perspective
    """

    def __init__(self):
        super(PatchTransformer_nonaff, self).__init__()
        self.min_contrast = 0.80
        self.max_contrast = 1
        self.min_brightness = -0.2
        self.max_brightness = 0
        self.noise_factor = 0.18
        self.minangle = -30 / 180 * math.pi
        self.maxangle = 30 / 180 * math.pi

        self.shminangle = -25 / 180 * math.pi
        self.shmaxangle = 25 / 180 * math.pi

        self.minscaleH = 1.5
        self.maxscaleH = 2

        self.minscaleW = 1.5
        self.maxscaleW = 2

        self.transformerset = torch.nn.Sequential(
            # transforms.ColorJitter(saturation=(0.6,1)),
            transforms.RandomAdjustSharpness(sharpness_factor=0, p=1),
        )

        self.medianpooler = MedianPool2d(7, same=True)

        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # Contrast, brightness and noise transforms

        # Apply gaussian blur (abandon)
        size0 = adv_batch.size(0)
        size1 = adv_batch.size(1)
        size3 = adv_batch.size(3)
        size4 = adv_batch.size(4)
        adv_batch = adv_batch.view(-1, 3, size3, size4)

        adv_batch = self.transformerset(adv_batch)
        adv_batch = adv_batch.view(size0, size1, 3, size3, size4)


        adv_batch = torch.clamp(adv_batch, 0.03, 0.95)



        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise


        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # print(lab_batch)
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids[:, 0, 0] = 0
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        # print(cls_mask)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            shearangle_x = torch.cuda.FloatTensor(anglesize).uniform_(self.shminangle, self.shmaxangle)
            shearangle_y = torch.cuda.FloatTensor(anglesize).uniform_(self.shminangle, self.shmaxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05

        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        hightscaleindex = torch.cuda.FloatTensor(anglesize).uniform_(self.minscaleH, self.maxscaleH)
        wscaleindex = torch.cuda.FloatTensor(anglesize).uniform_(self.minscaleW, self.maxscaleW)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        sheartanhx = torch.tanh(shearangle_x)
        sheartanhy = torch.tanh(shearangle_y)

        ''''''
        # Theta = rotation,rescale matrix

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale



        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid,align_corners = True)
        msk_batch_t = F.grid_sample(msk_batch, grid,align_corners = True)
        # ''''''


        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.03, 0.95)


        return adv_batch_t * msk_batch_t



class PatchTransformer_myaff(nn.Module):
    """PatchTransformer: random pers. and random affine

    """

    def __init__(self):
        super(PatchTransformer_myaff, self).__init__()
        self.transformerset1 = torch.nn.Sequential(
            transforms.ColorJitter(brightness=(0.8,1.1),contrast=(0.8,1.1),saturation=(0.8,1.1)),
            # transforms.GaussianBlur(kernel_size=(1,21),sigma=(0.01,3)),
        )

        self.transformerset2 = torch.nn.Sequential(
            transforms.RandomAffine(degrees=(-45, 45), scale=(0.10, 0.16)),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        )
        self.noise_factor = 0.15

        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):


        adv_patch = adv_patch.unsqueeze(0).unsqueeze(0)  # .unsqueeze(0)

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)


        size0 = adv_batch.size(0)
        size1 = adv_batch.size(1)
        size3 = adv_batch.size(3)
        size4 = adv_batch.size(4)
        adv_batch = adv_batch.view(-1, 3, size3, size4)

        adv_batch = transforms.Resize(size=img_size)(adv_batch)
        adv_batch = self.transformerset1(adv_batch)
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor


        adv_batch = adv_batch + noise
        adv_batch = self.transformerset2(adv_batch)
        adv_batch = adv_batch.view(size0, size1, 3, img_size, img_size)
        adv_batch = torch.clamp(adv_batch, 0, 0.99)


        return adv_batch



class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv ==0), img_batch, adv)
        return img_batch

'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)

        transform = transforms.Compose([
            transforms.Resize(self.imgsize),
            transforms.RandomRotation(degrees=30, fill=25),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5),
            transforms.TrivialAugmentWide(fill=25),
            # transforms.ColorJitter(saturation=(0.6,1.4)),
            # transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])

        image = transform(image)
        label = self.pad_lab(label)

        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(25,25,25))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(25,25,25))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab