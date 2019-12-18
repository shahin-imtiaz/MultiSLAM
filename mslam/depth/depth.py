# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(1, '/home/userm/myGithub/MultiSLAM/monodepth2_repo')

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

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist




# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing funtion for Monodepthv2 models.')

#     parser.add_argument('--image_path', type=str,
#                         help='path to a test image or folder of images', required=True)
#     parser.add_argument('--model_name', type=str,
#                         help='name of a pretrained model to use',
#                         choices=[
#                             "mono_640x192",
#                             "stereo_640x192",
#                             "mono+stereo_640x192",
#                             "mono_no_pt_640x192",
#                             "stereo_no_pt_640x192",
#                             "mono+stereo_no_pt_640x192",
#                             "mono_1024x320",
#                             "stereo_1024x320",
#                             "mono+stereo_1024x320"])
#     parser.add_argument('--ext', type=str,
#                         help='image extension to search for in folder', default="jpg")
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')

#     return parser.parse_args()

class MonoDepth:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model_path = os.path.join("monodepth2_repo/models", "mono_1024x320")
        # print("-> Loading model from ", self.model_path)
        self.encoder_path = os.path.join(self.model_path, "encoder.pth")
        self.depth_decoder_path = os.path.join(self.model_path, "depth.pth")
        
        # LOADING PRETRAINED MODEL
        # print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def estimate(self, img):
        with torch.no_grad():
            # for idx, image_path in enumerate(paths):

            #     if image_path.endswith("_disp.jpg"):
            #         # don't try to predict disparity for a disparity image!
            #         continue

            #     # Load image and preprocess
            #     img = pil.open(image_path).convert('RGB')
            img = Image.fromarray(img).convert('RGB')
            original_width, original_height = img.size
            img = img.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            img = transforms.ToTensor()(img).unsqueeze(0)

            # PREDICTION
            img = img.to(self.device)
            features = self.encoder(img)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth imaged
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            
            return colormapped_im
            # im = pil.fromarray(colormapped_im)

            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            # print("   Processed {:d} of {:d} images - saved prediction to {}".format(
            #     idx + 1, len(paths), name_dest_im))

    # print('-> Done!')


# def test_simple(args):

    # # FINDING INPUT IMAGES
    # if os.path.isfile(args.image_path):
    #     # Only testing on a single image
    #     paths = [args.image_path]
    #     output_directory = os.path.dirname(args.image_path)
    # elif os.path.isdir(args.image_path):
    #     # Searching folder for images
    #     paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    #     output_directory = args.image_path
    # else:
    #     raise Exception("Can not find args.image_path: {}".format(args.image_path))

    # print("-> Predicting on {:d} test images".format(len(paths)))



if __name__ == '__main__':
    md = MonoDepth()
    im = cv2.imread("input1.jpg")
    cv2.imwrite("out.jpg", md.estimate(im))