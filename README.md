# MultiSLAM
Enhanced Real-Time Multi-Camera SLAM for 3D Scene Reconstruction
By: Mansoor Saqib & Shahin Imtiaz
For: University of Toronto CSC420 - Intro to Image Understanding Fall 2019

## Running on CDF

> $ bash mslam_setup_cdf.sh
> $ python3 mslam.py

## Personal Dev Environment Setup
>   $ sudo apt-get install python-dev python3-dev
>
>   $ python3 -m venv env-mslam
>
>   $ source env-mslam/bin/activate
>
>   (env-mslam) $ python3 -m pip install -U pip
>
>   (env-mslam) $ python3 -m pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
>
>   (env-mslam) $ python3 -m pip install -r requirements.pip
>
>   (env-mslam) $ git clone https://github.com/facebookresearch/detectron2 detectron2_repo
>
>   (env-mslam) $ python3 -m pip install -e detectron2_repo

## Package contents
* [mslam/main.py](mslam/main.py): Entrypoint to run MultiSLAM
* [mslam/agent_loc](mslam/agent_loc):
    - [agent_loc.py](mslam/agent_loc/agent_loc.py): Determine agent movement in 3D space
* [mslam/depth](mslam/depth):
    - [mono.py](mslam/depth/depth.py): Compute mono depth information from a single image
    - [stereo_cnn.py](mslam/depth/depth.py): Compute stereo depth information from a two images
* [mslam/geo_proj](mslam/geo_proj):
    - [geo_proj.py](mslam/geo_proj/geo_proj.py): Construct the scene point cloud and track agent location
* [mslam/img_stitch](mslam/img_stitch):
    - [img_stitch.py](mslam/img_stitch/img_stich.py): Stitch together a set of images
* [mslam/obj_rec](mslam/obj_rec):
    - [obj_rec.py](mslam/obj_rec/obj_rec.py): Classify the objects in a single or set of images