# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(1, '../../monodepth2_repo')

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import torch
from torchvision import transforms, datasets

import monodepth2_repo.networks as networks
from monodepth2_repo.layers import disp_to_depth
from monodepth2_repo.utils import download_model_if_doesnt_exist

'''
Monocular Depth
Utilizes the Monodepth2 CNN: https://github.com/nianticlabs/monodepth2
Creates a depth map from a single frame
NOTE: Makes use of the variable naming and calling conventions found in the library's predictor script
'''
class MonoDepth:
    # Initialize the predictor
    def __init__(self, path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Load model
        self.model_path = path
        self.encoder_path = os.path.join(self.model_path, "encoder.pth")
        self.depth_decoder_path = os.path.join(self.model_path, "depth.pth")
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        # Model training paramaters
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(self.loaded_dict)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def estimate(self, img):
        with torch.no_grad():
            # Convert cv2 image array to PIL Image
            img = Image.fromarray(img).convert('RGB')
            original_width, original_height = img.size
            img = img.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            img = transforms.ToTensor()(img).unsqueeze(0)

            # Get prediction
            img = img.to(self.device)
            features = self.encoder(img)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Generate depth map
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            return colormapped_im