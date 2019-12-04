# Flask streaming webserver based on the guide from:
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

from mslam.geo_proj.geo_proj import GeoProjection
from mslam.obj_rec.objectdetect import ObjectDetect
from mslam.depth.depth import MonoDepth
# from mslam.depth.depth_hd3 import StereoDepth TODO
from mslam.agent_loc.agent_loc import AgentLocate

from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

debugging = True
isLiveFeed = False

enableModules = {
    'object_detection': True,
    'stereo_depth': False,
    'agent_locate': False,
    'mono_depth': True,     # <----
    'geo_projection': False, # -----^ dependency
}

outputFrame = {
    'object_detection': None,
    'mono_depth': None,
    'stereo_depth': None,
    'agent_locate': None,
    'geo_projection': None
}

app = Flask(__name__)

vs = None   # Left camera stream
vs2 = None  # Right camera stream

@app.route("/")
def index():
	return render_template("index.html")

def mslam():
    global vs, enableModules, outputFrame
    
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
        mod_geo_projection = GeoProjection()

    if debugging:
        print("Done initializing modules")
        print("Starting MultiSlam rendering")

    while True:
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
            mod_geo_projection.estimate(frame.copy(), outputFrame['mono_depth'].copy(), downsample=1000)

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

@app.route("/stream_object_detection")
def stream_object_detection():
	return Response(generateStreamFrame('object_detection'),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_mono_depth")
def stream_mono_depth():
	return Response(generateStreamFrame('mono_depth'),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_stereo_depth")
def stream_stereo_depth():
	return Response(generateStreamFrame('stereo_depth'),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_agent_locate")
def stream_agent_locate():
	return Response(generateStreamFrame('agent_locate'),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# NOTE: No stream for geo_projection as it is rendered in the Open3D visualizer

if __name__ == '__main__':
    # Arg parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address")
    ap.add_argument("-o", "--port", type=int, required=True, help="port number")
    ap.add_argument("-L", "--leftcam", type=str, required=True, help="left camera stream")
    ap.add_argument("-R", "--rightcam", type=str, required=False, help="right camera stream")
    args = vars(ap.parse_args())
 
    # Camera feed setup
    vs = cv2.VideoCapture(args["leftcam"])
    if args["rightcam"] is not None:
        vs2 = cv2.VideoCapture(args["rightcam"])

    if isLiveFeed:
        time.sleep(2.0)

    # Single threaded multislam
    t = threading.Thread(target=mslam)
    t.daemon = True
    t.start()
 
    # Start flask webserver
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
