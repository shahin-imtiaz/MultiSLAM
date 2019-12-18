# Based on the UCBDrive's HD3 libraries
# https://github.com/ucbdrive/hd3
#
# Currently work is still in progress to make this module functional as there are a few library errors

from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(1, '../../hd3_repo')
sys.path.insert(1, '../../hd3_repo/utils')

import os
from os.path import join
import cv2
import time
import math
import logging
from argparse import ArgumentParser
import numpy as np

from PIL import Image

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# import data.hd3data as datasets
# import data.flowtransforms as transforms
# import hd3model as models
# from utils.utils import *
# from models.hd3_ops import *
# import utils.flowlib as fl

import hd3_repo.data.hd3data as datasets
import hd3_repo.data.flowtransforms as transforms
import hd3_repo.hd3model as models
from hd3_repo.utils.utils import *
from hd3_repo.models.hd3_ops import *
import hd3_repo.utils.flowlib as fl

'''
Stereo Depth
Utilizes the HD3 CNN: https://github.com/ucbdrive/hd3
Creates a depth map from two stereoscopic frames
NOTE: Currently a work in progress
NOTE: Makes use of the variable naming and calling conventions found in the library's predictor script
'''
class StereoDepthCNN:
    def __init__(self, model_path):
        self.corr_range = [4, 4, 4, 4, 4, 4]
        self.model = models.HD3Model("stereo", "dlaup", "hda", self.corr_range,
                                False).cuda()
        self.model = torch.nn.DataParallel(self.model).cuda()
        
        self.checkpoint = torch.load(model_path)
        self.model.load_state_dict(self.checkpoint['state_dict'], strict=True)
        
        self.model.eval()

    def get_target_size(self, H, W):
        h = 64 * np.array([[math.floor(H / 64), math.floor(H / 64) + 1]])
        w = 64 * np.array([[math.floor(W / 64), math.floor(W / 64) + 1]])
        ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - H / W)
        index = np.argmin(ratio)
        return h[0, index // 2], w[0, index % 2]

    def estimate(self, imgL, imgR):
        input_size = imgL.shape

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        th, tw = self.get_target_size(input_size[0], input_size[1])
        
        val_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    
        mode="stereo"
        label_num=False
        transform=val_transform
        out_size=True

        cudnn.enabled = True
        cudnn.benchmark = True

        cur_data = [[imgL, imgR], []]
        cur_data = list(val_transform(*cur_data))
        cur_data.append(np.array(input_size[::-1], dtype=int))
        cur_data = tuple(cur_data)

        with torch.no_grad():
            img_list, label_list, img_size = cur_data
            img_list = [img.unsqueeze(0) for img in img_list]
            img_size = np.array(input_size[:2][::-1], dtype=int)
            img_size = img_size[np.newaxis, :]
            img_list = [img.to(torch.device("cuda")) for img in img_list]
            label_list = [
                label.to(torch.device("cuda")) for label in label_list
            ]

            # resize test
            resized_img_list = [
                F.interpolate(
                    img, (th, tw), mode='bilinear', align_corners=True)
                for img in img_list
            ]

            output = self.model(
                img_list=resized_img_list,
                label_list=label_list,
                get_vect=True,
                get_epe=False)
            scale_factor = 1 / 2**(7 - len(self.corr_range))
            output['vect'] = resize_dense_vector(output['vect'] * scale_factor,
                                                img_size[0, 1],
                                                img_size[0, 0])

            pred_vect = output['vect'].data.cpu().numpy()
            pred_vect = np.transpose(pred_vect, (0, 2, 3, 1))
            curr_vect = pred_vect[0]

            vis_flo = fl.flow_to_image(fl.disp2flow(curr_vect))
            vis_flo = cv2.cvtColor(vis_flo, cv2.COLOR_RGB2BGR)
            return vis_flo