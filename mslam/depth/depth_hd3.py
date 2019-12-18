from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(1, '/home/userm/myGithub/MultiSLAM/hd3_repo')

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

import data.hd3data as datasets
import data.flowtransforms as transforms
import hd3model as models
from utils.utils import *
from models.hd3_ops import *
import utils.flowlib as fl

class StereoDepth:
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

        # imgL = Image.fromarray(imgL).convert('RGB')
        # imgR = Image.fromarray(imgR).convert('RGB')
        # imgL = Image.fromarray(imgL)
        # imgR = Image.fromarray(imgR)
        # print(imgR.size)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        th, tw = self.get_target_size(input_size[0], input_size[1])
        # print(th, tw)
        
        val_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
        # val_data = datasets.HD3Data(
        mode="stereo"
        # data_root=args.data_root,
        # data_list=args.data_list,
        label_num=False
        transform=val_transform
        out_size=True
            # )
        
        # val_loader = torch.utils.data.DataLoader(
        #     val_data,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=16,
        #     pin_memory=True)

        cudnn.enabled = True
        cudnn.benchmark = True
        
        # imgL = val_transform(imgL, [])
        # imgR = val_transform(imgR, [])
        # print(imgL.shape)
        cur_data = [[imgL, imgR], []]
        cur_data = list(val_transform(*cur_data))
        cur_data.append(np.array(input_size[::-1], dtype=int))
        cur_data = tuple(cur_data)

        # print(cur_data)
        # imgL_tensor = cur_data[0][0]
        # print(imgL)
        # imgR_tensor = cur_data[0][1]

        with torch.no_grad():
            # for i, (img_list, label_list, img_size) in enumerate(val_loader):
                # data_time.update(time.time() - end)

            img_list, label_list, img_size = cur_data
            # print(img_list)
            img_list = [img.unsqueeze(0) for img in img_list]
            img_size = np.array(input_size[:2][::-1], dtype=int)
            img_size = img_size[np.newaxis, :]
            # print(img_size)
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



            # # img_size = np.array(imgL.size, dtype=int).cpu().numpy()
            # img_size = np.array(imgL.size, dtype=int)
            # img_list = [imgL_tensor.to(torch.device("cuda")), imgR_tensor.to(torch.device("cuda"))]
            # label_list = []
            # print(img_list[0].shape)

            # # resize test
            # resized_img_list = [
            #     F.interpolate(
            #         img, (th, tw), mode='bilinear', align_corners=True)
            #     for img in img_list
            # ]


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
            # curr_bs = pred_vect.shape[0]
            # assert curr_bs == 1

            # for idx in range(curr_bs):
                # curr_idx = i * 1 + idx
            curr_vect = pred_vect[0]

            vis_flo = fl.flow_to_image(fl.disp2flow(curr_vect))
            vis_flo = cv2.cvtColor(vis_flo, cv2.COLOR_RGB2BGR)
            return vis_flo

                    # cv2.imwrite(vis_fn, vis_flo)

                    # cv2.imwrite(vect_fn,
                    #             np.uint16(-curr_vect[:, :, 0] * 256.0))

       

if __name__ == '__main__':
    sd = StereoDepth("../../hd3_repo/scripts/model_zoo/hd3s_things_kitti-1243813e.pth")
    imL = cv2.imread("imgL.png")
    imR = cv2.imread("imgR.png")
    cv2.imwrite("outStereo.png", sd.estimate(imL, imR))