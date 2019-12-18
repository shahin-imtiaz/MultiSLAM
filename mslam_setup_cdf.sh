#!/bin/bash

echo "Installing mslam"
export CUDA_HOME='/usr/local/cuda-10.0/'
python3 -m venv env-mslam
source env-mslam/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.pip
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
git clone https://github.com/nianticlabs/monodepth2.git
git clone https://github.com/ucbdrive/hd3.git
python3 -m pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
python3 -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python3 -m pip install -e detectron2_repo
python3 -m pip install -U "git+https://github.com/cocodataset/panopticapi.git"
python3 -m pip install -U "git+https://github.com/lvis-dataset/lvis-api.git"
python3 -m pip install -U "pip install git+https://github.com/mcordts/cityscapesScripts.git"
python3 -m pip install -U "git+https://github.com/mcordts/cityscapesScripts.git"
python3 -m pip install -r requirements.pip
cd monodepth2_repo && python3 test_simple.py --image_path assets/test_image.jpg --model_name mono_1024x320
cd ../
echo "Done setup. Please run: python3 mslam.py"