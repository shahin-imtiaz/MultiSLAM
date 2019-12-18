# Flask streaming webserver based on the guide from:
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

from mslam.geo_proj.geo_proj import GeoProjection
from mslam.obj_rec.objectdetect import ObjectDetect
from mslam.depth.depth import MonoDepth
# from mslam.depth.depth_hd3 import StereoDepth TODO
from mslam.agent_loc.agent_loc import AgentLocate

import threading
import argparse
import datetime
import imutils
import time
import cv2
from tqdm import tqdm
import numpy as np
import open3d as o3d

debugging = False
length = 0

enableModules = {
    'object_detection': True,
    'stereo_depth': False,
    'agent_locate': False,
    'mono_depth': True,         # <----
    'geo_projection': False,     # -----^ dependency
}

outputFrame = {
    'object_detection': None,
    'mono_depth': None,
    'stereo_depth': None,
    'agent_locate': None,
    'geo_projection': None
}

outputWriter = {}

vs = None   # Left camera stream
vs2 = None  # Right camera stream

def mslam():
    global vs, enableModules, outputFrame, length
    
    # Initialize modules
    mod_object_detection = None
    mod_mono_depth = None
    mod_stereo_depth = None
    mod_agent_locate = None
    mod_geo_projection = None

    if debugging:
        print("Initializing modules")

    if enableModules['object_detection']:
        mod_object_detection = ObjectDetect("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if enableModules['mono_depth']:
        mod_mono_depth = MonoDepth()
    if enableModules['stereo_depth']:
        mod_stereo_depth = StereoDepth("hd3_repo/scripts/model_zoo/hd3s_things_kitti-1243813e.pth")
    if enableModules['agent_locate']:
        mod_agent_locate = AgentLocate()
    if enableModules['geo_projection']:
        mod_geo_projection = GeoProjection(mode='offline')

    if debugging:
        print("Done initializing modules")
        print("Starting MultiSlam rendering")

    for i in tqdm(range(length)):
        # Get next frame
        isValid, frame = vs.read()
        if not isValid: # No more frames
            break
        
        # Get next frame for right camera
        if enableModules['stereo_depth']:
            isValid, frame2 = vs2.read()
            if not isValid:
                break

        if enableModules['mono_depth']:
            outputFrame['mono_depth'] = mod_mono_depth.estimate(frame.copy())
        if enableModules['agent_locate']:
            outputFrame['agent_locate'] = mod_agent_locate.estimate(frame.copy())
        if enableModules['object_detection']:
            outputFrame['object_detection'] = mod_object_detection.estimate(frame.copy())
        if enableModules['geo_projection']:
            outputFrame['geo_projection'] = mod_geo_projection.estimate(frame.copy(), outputFrame['mono_depth'].copy(), downsample=1000)
        #print(hash(np.sum(outputFrame['object_detection'])))
        writeFrame(i)

def writeFrame(frameNum):
    global outputFrame, outputWriter, enableModules, debugging

    for m in enableModules.keys():

        if not enableModules[m] or outputFrame[m] is None:
            continue

        if m == 'geo_projection':
            o3d.io.write_point_cloud(outputWriter[m], outputFrame['geo_projection'])
            continue

        if debugging:
            print("writing frame", frameNum, "for", m)
            print("frame stats: ", np.max(outputFrame[m]), np.min(outputFrame[m]), outputFrame[m].shape)

        outputWriter[m].write(outputFrame[m])

# NOTE: No stream for geo_projection as it is rendered in the Open3D visualizer

if __name__ == '__main__':
    # Arg parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-L", "--leftcam", type=str, required=True, help="left camera stream")
    ap.add_argument("-R", "--rightcam", type=str, required=False, help="right camera stream")
    ap.add_argument("-O", "--output", type=str, required=True, help="output path")
    ap.add_argument("-e", "--endframe", type=int, required=False, help="end frame number")
    args = vars(ap.parse_args())
 
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

    # Ouput format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output writers
    for m in enableModules.keys():
        if enableModules[m]:
            if m is 'geo_projection':
                outputWriter[m] = args["output"]+'OUT_' + m +'.pcd'
            else:
                outputWriter[m] = cv2.VideoWriter(args["output"]+'OUT_' + m +'.mp4',fourcc,30,(width,height))

    # Single threaded multislam
    t = threading.Thread(target=mslam)
    t.daemon = True
    t.start()
    
    # Wait for thread to finish
    t.join()
