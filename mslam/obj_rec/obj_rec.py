# Makes use of https://github.com/facebookresearch/detectron2/
# Detectron2 Tutorial (colab):
#    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0

import torch, torchvision
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pprint

'''
Object Detection
Utilizes Facebook AI Research's Detectron2 CNN: https://github.com/facebookresearch/detectron2
Performs frame by frame object detection and segmentation
NOTE: Makes use of the variable naming and calling conventions found in the library's predictor script
'''
class ObjectDetect:
    # Initialize the predictor
    def __init__(self, model_yaml, weights):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_yaml)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = weights
        self.predictor = DefaultPredictor(self.cfg)
    
    # Feed in an image frame to the predictor and return the output
    def estimate(self, img):
        outputs = self.predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]
    
    # Return the current CfgNode
    def getCFG(self):
        return self.cfg