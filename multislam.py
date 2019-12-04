# Based on the guide from https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

from mslam.geo_proj.geo_proj import PointCloud
# import the necessary packages
from mslam.obj_rec.objectdetect import ObjectDetect
from mslam.depth.depth import MonoDepth
# from mslam.depth.depth_hd3 import StereoDepth

# from mslam.geo_proj.geo_proj import PointCloud
from imutils.video import VideoStream, FileVideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

outputObjDetFrame = None
outputDepthFrame = None
outputStereoDepthFrame = None
outputSIFTFrame = None

lock = threading.Lock()
app = Flask(__name__)
# vs = VideoStream(src=0).start()
# vs = FileVideoStream("video/Back to school, University of Toronto St. George Campus walk-U_cLTtisR0s.mp4")
# vs.start()
# time.sleep(2.0)
vs = cv2.VideoCapture("video/class2mp4.mp4")


@app.route("/")
def index():
	return render_template("index.html")

def mslam_obj_detect():
    global vs, outputObjDetFrame, outputDepthFrame, outputStereoDepthFrame, outputSIFTFrame, lock
    od = ObjectDetect("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    md = MonoDepth()
    # sd = StereoDepth("hd3_repo/scripts/model_zoo/hd3s_things_kitti-1243813e.pth")
    sd_prev_frames = []
    # proc_sift = PointCloud()
    pc3d = PointCloud()

    while True:
        print('inside detect')
        gotFrame, frame = vs.read()
        if not gotFrame:
            break
        print('inside detect2')
        # frame = imutils.resize(frame, width=300)
        # height, width = frame.shape[:2]
        # print(frame.shape)
        # frame = cv2.resize(frame, (int(0.2*width), int(0.2*height)))
        # print(frame.shape)
        print('inside detect3')
        # outputSIFTFrame = proc_sift.estimate(frame.copy())
        outMonoDepth = md.estimate(frame.copy())
        pc3d.estimate(frame.copy(), outMonoDepth.copy(), downsample=1000)
        # if len(sd_prev_frames) < 20:
        #     sd_prev_frames.append(frame.copy())
        #     continue
        # else:
        #     sd_prev_frames.append(frame.copy())
        #     outStereoDepth = sd.estimate(sd_prev_frames.pop(0), frame)
        outObjDet = od.detect(frame)
        # out = frame
        print('outputting frame')
        # with lock:
        # cv2.imwrite("coolcool.jpg", frame)
        outputObjDetFrame = outObjDet.copy()
        outputDepthFrame = outMonoDepth.copy()
        # outputStereoDepthFrame = outStereoDepth.copy()
        # time.sleep(2)

def generateObjDet():
    global outputObjDetFrame, lock

    while True:
        with lock:
            if outputObjDetFrame is None:
                continue
 
            (flag, encodedImage) = cv2.imencode(".jpg", outputObjDetFrame)
 
            if not flag:
                continue
 
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def generateMonoDepth():
    global outputDepthFrame, lock

    while True:
        with lock:
            if outputDepthFrame is None:
                continue
 
            (flag, encodedImage) = cv2.imencode(".jpg", outputDepthFrame)
 
            if not flag:
                continue
 
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def generateStereoDepth():
    global outputStereoDepthFrame, lock

    while True:
        with lock:
            if outputStereoDepthFrame is None:
                continue
 
            (flag, encodedImage) = cv2.imencode(".jpg", outputStereoDepthFrame)
 
            if not flag:
                continue
 
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

def generateSIFT():
    global outputSIFTFrame, lock

    while True:
        with lock:
            if outputSIFTFrame is None:
                continue
 
            (flag, encodedImage) = cv2.imencode(".jpg", outputSIFTFrame)
 
            if not flag:
                continue
 
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed_obj_det")
def video_feed_obj_det():
	return Response(generateObjDet(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_mono_depth")
def video_feed_mono_depth():
	return Response(generateMonoDepth(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_stereo_depth")
def video_feed_stereo_depth():
	return Response(generateStereoDepth(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_sift")
def video_feed_sift():
	return Response(generateSIFT(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())
 
	# start a thread that will perform motion detection
	t = threading.Thread(target=mslam_obj_detect)
	t.daemon = True
	t.start()
 
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
 
# release the video stream pointer
vs.stop()