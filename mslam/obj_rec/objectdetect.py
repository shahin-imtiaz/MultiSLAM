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

# setup_logger()
# pp = pprint.PrettyPrinter(indent=2)

class ObjectDetect:
    def __init__(self, model_yaml):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_yaml)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.predictor = DefaultPredictor(self.cfg)
    
    def estimate(self, img):
        outputs = self.predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]
    
    def getCFG(self):
        return self.cfg
        

if __name__ == "__main__":
    od = ObjectDetect("../../detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    od.getCFG
    im = cv2.imread("input1.jpg")
    cv2.imwrite("out.jpg", od.detect(im))
    # pp.pprint(dir(models))


