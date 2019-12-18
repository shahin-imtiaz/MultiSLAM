##CSC420 Fall 2019 Final Project - MultiSLAM
##By: Mansoor Saqib & Shahin Imtiaz

from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(1, 'hd3_repo')
sys.path.insert(1, 'monodepth2_repo')

import os
from os.path import join
import threading
import argparse
import datetime
import imutils
import time
import copy
import cv2
import logging
import pprint
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import open3d as o3d
import PIL.Image as pil
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats

from mslam.agent_loc.agent_loc import AgentLocate
from mslam.depth.mono import MonoDepth
from mslam.geo_proj.geo_proj import GeoProjection
# from mslam.depth.stereo_CNN import StereoDepthCNN # Stereo Depth currently not working
from mslam.obj_rec.obj_rec import ObjectDetect

############################
####   MultiSlam Loop   ####
############################

# The main rendering loop. Combines the SLAM modules together to create
# a mapping of the provided environment video or real time file stream.
def mslam():
    global vs, enableModules, outputFrame, length, settingsModules, debugging, live_stream
    
    # Initialize modules
    mod_object_detection = None
    mod_mono_depth = None
    mod_stereo_depth_cnn = None
    mod_agent_locate = None
    mod_geo_projection = None

    if debugging:
        print("Initializing modules")

    if enableModules['object_detection']:
        mod_object_detection = ObjectDetect(
            settingsModules['object_detection']['model_path'],
            settingsModules['object_detection']['model_weights'])
    if enableModules['mono_depth']:
        mod_mono_depth = MonoDepth(settingsModules['mono_depth']['model_path'])
    if enableModules['stereo_depth_cnn']:
        mod_stereo_depth_cnn = StereoDepthCNN(settingsModules['stereo_depth_cnn']['model_path'])
    if enableModules['agent_locate']:
        mod_agent_locate = AgentLocate(debugging=debugging)
    if enableModules['geo_projection']:
        if live_stream:
            mod_geo_projection = GeoProjection(mode='online')
        else:
            mod_geo_projection = GeoProjection(mode='offline')

    if debugging:
        print("Done initializing modules")
        print("Starting MultiSlam rendering")

    # Loop through video frames
    for i in tqdm(range(length)):

        # Get next frame
        isValid, frame = vs.read()
        if not isValid: # No more frames
            break
        outputFrame['original_L'] = frame.copy()
        
        # Get next frame for right camera
        if enableModules['stereo_depth_cnn']:
            isValid, frame2 = vs2.read()
            if not isValid:
                break
            outputFrame['original_R'] = frame2.copy()

        # Feed in frames to modules and obtain output
        if enableModules['mono_depth']:
            outputFrame['mono_depth'] = mod_mono_depth.estimate(frame.copy())
        if enableModules['stereo_depth_cnn']:
            outputFrame['stereo_depth_cnn'] = mod_stereo_depth_cnn.estimate(frame.copy())
        if enableModules['agent_locate']:
            out_agent_locate = mod_agent_locate.estimate(frame.copy())
            outputFrame['agent_locate'] = out_agent_locate['frame']
            al_transform = out_agent_locate['transform']
        if enableModules['object_detection']:
            outputFrame['object_detection'] = mod_object_detection.estimate(frame.copy())
        if enableModules['geo_projection']:
            outPCD = mod_geo_projection.estimate(frame.copy(), outputFrame['mono_depth'].copy(), al_transform, downsample=10)
            outputFrame['geo_projection_pcd'] = outPCD[0]
            outputFrame['geo_projection'] = outPCD[1]
        
        # Write frame to current point cloud or video file
        writeFrame(i)
    
    print("Done processing video. Output has been saved in:", args['output'])
    print("Closing mslam")
    time.sleep(10)
    print("Please exit the application with CTRL+C")
    exit(0)

# Write the current frame among all modules to a file. Alternatively, this
# can be adjusted to have the frame streamed for a live view as rendering is in progress.
## frameNum = id of the frame being processed
def writeFrame(frameNum):
    global outputFrame, outputWriter, enableModules, debugging

    for m in enableModules.keys():
        if not enableModules[m] or outputFrame[m] is None:
            continue

        if m == 'geo_projection_pcd' or m == 'geo_projection':
            if debugging:
                print("Adding points in frame", frameNum, "to global point cloud")
                print("writing frame", frameNum, "for", m)
                print("frame stats: ", np.max(outputFrame[m]), np.min(outputFrame[m]), outputFrame[m].shape)
            o3d.io.write_point_cloud(outputWriter['geo_projection_pcd'], outputFrame['geo_projection_pcd'])
            outputWriter['geo_projection'].write(outputFrame['geo_projection'])
            continue

        if debugging:
            print("writing frame", frameNum, "for", m)
            print("frame stats: ", np.max(outputFrame[m]), np.min(outputFrame[m]), outputFrame[m].shape)

        outputWriter[m].write(outputFrame[m])

# Contains the current frame output for each module (or point cloud for geo_projection)
outputFrame = {
    'object_detection': None,
    'mono_depth': None,
    'stereo_depth_cnn': None,
    'agent_locate': None,
    'geo_projection': None,
    'geo_projection_pcd': None,
    'original_L': None,
    'original_R': None,
}

############################
#### User Configuration ####
################################################################################################

print('MSLAM options can be configured in mslam.py under User Configuration')

# Initial configuration for input and output rendering settings
args = {
    'leftcam': 'video/kittbw.mp4',          # Path to main or left camera video
    # 'leftcam': 'video/driving_country_480p_trimmed.mp4',
    'rightcam': None,                       # Path to right camera video if stereo is enabled
    'output': 'OUTPUT/',                    # Path to rendering output
    'endframe': 1400,                         # Total number of video frames to process. None = All
    'ip': '0.0.0.0',                        # Address for live streaming. Can access with VLC network stream
    'port': '8000',                         # Port for live streaming. Can access with VLC network stream
}

# Verbose execution
debugging = False

# Live stream the output
live_stream = True

# Enable or disable the modules for debugging. For a complete slam system, enable all* (*choose one of mono or stereo depth).
enableModules = {
    'object_detection': True,
    'stereo_depth_cnn': False,
    'agent_locate': True,
    'mono_depth': True,
    'geo_projection': True,     # depends on depth and agent_locate
}

# Module specific settings
settingsModules = {
    'object_detection': {
        'model_path': "detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        'model_weights': "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
        },
    'stereo_depth_cnn': {
        'model_path': "hd3_repo/scripts/model_zoo/hd3s_things_kitti-1243813e.pth",
        },
    'agent_locate': {},
    'mono_depth': {
        'model_path': os.path.join("monodepth2_repo/models", "mono_1024x320"),
        },
    'geo_projection': {},
}


################################################################################################
####    Run MultiSlam   ####
############################

# Camera feed setup
vs = cv2.VideoCapture(args["leftcam"])
if args["rightcam"] is not None:
    vs2 = cv2.VideoCapture(args["rightcam"])

# Get video stats
length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

if args["endframe"] is not None:
    length = args["endframe"]

print("MSLAM will run for "+str(length)+" frames")

# Ouput format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Output writers
outputWriter = {}
for m in enableModules.keys():
    if enableModules[m]:
        if m is 'geo_projection':
            outputWriter['geo_projection_pcd'] = args["output"]+'OUT_' + 'geo_projection' +'.pcd'
            outputWriter['geo_projection'] = cv2.VideoWriter(args["output"]+'OUT_' + 'geo_projection' +'.mp4',fourcc,30,(1850,1016))
        else:
            outputWriter[m] = cv2.VideoWriter(args["output"]+'OUT_' + m +'.mp4',fourcc,30,(width,height))

# Single threaded multislam
t = threading.Thread(target=mslam)
t.daemon = True
t.start()

# Video streaming the output with the Flask framework is based on
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
if live_stream:
    from flask import Response
    from flask import Flask
    from flask import render_template
    
    app = Flask(__name__)

    def generateStreamFrame(moduleName):
        global outputFrame

        while True:
            if outputFrame[moduleName] is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame[moduleName])

            if not flag:
                continue
    
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

    @app.route("/multislam_stream_original_L")
    def stream_original_L():
        return Response(generateStreamFrame('original_L'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_original_R")
    def stream_original_R():
        return Response(generateStreamFrame('original_R'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_object_detection")
    def stream_object_detection():
        return Response(generateStreamFrame('object_detection'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_mono_depth")
    def stream_mono_depth():
        return Response(generateStreamFrame('mono_depth'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_stereo_depth_cnn")
    def stream_stereo_depth_cnn():
        return Response(generateStreamFrame('stereo_depth_cnn'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_agent_locate")
    def stream_agent_locate():
        return Response(generateStreamFrame('agent_locate'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/multislam_stream_geo_projection")
    def stream_geo_projection():
        return Response(generateStreamFrame('geo_projection'),
            mimetype = "multipart/x-mixed-replace; boundary=frame")
    
    app.run(host=args["ip"], port=args["port"], debug=False, threaded=True, use_reloader=False)

# Wait for thread to finish
t.join()