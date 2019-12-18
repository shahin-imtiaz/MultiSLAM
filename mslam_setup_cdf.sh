#!/bin/bash

set -eu -o pipefail

echo "Installing mslam. This computer must have CUDA enabled support"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-10.0/}"
python3 -m venv --system-site-packages env-mslam
source env-mslam/bin/activate
alias pip="python3 -m pip"
pip install -U pip
pip install -r requirements.pip
[ ! -f detectron2_repo ] && git clone https://github.com/facebookresearch/detectron2 detectron2_repo || true
[ ! -f monodepth2_repo ] && git clone https://github.com/nianticlabs/monodepth2.git monodepth2_repo || true
[ ! -f hd3_repo ] && git clone https://github.com/ucbdrive/hd3.git || true
pip install Cython cupy
pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install -e detectron2_repo
pip install -U "git+https://github.com/cocodataset/panopticapi.git"
pip install -U "git+https://github.com/lvis-dataset/lvis-api.git"
pip install -U "git+https://github.com/mcordts/cityscapesScripts.git"
pip install -r requirements.pip
cd monodepth2_repo && python3 test_simple.py --image_path assets/test_image.jpg --model_name mono_1024x320
cd ../
echo "Done setup. Please run: python3 mslam.py"
