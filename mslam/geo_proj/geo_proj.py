# Based on the Open3D libraries
# https://github.com/intel-isl/Open3D
#
# Customized with our own 3D position estimation of agent in point cloud

import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

'''
Geometric Projection
Utilizes the Open3D 3D Processing Library: https://github.com/intel-isl/Open3D
Creates a point cloud map of the region the agent is located in
'''
class GeoProjection():
    # Initialize an empty point cloud.
    def __init__(self, mode='offline'):
        self.mode = mode
        self.pcd = None
        self.xyz = np.zeros(3, dtype='float64')
        self.rot = np.zeros((3,1), dtype='float64')
        self.vis = None

        # If using live mode, create the visualizer window.
        if self.mode == 'online':
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
    
    # Fix view reset from https://github.com/intel-isl/Open3D/issues/497
    def center_view(self, vis):
        vis.reset_view_point(True)
        ctr = vis.get_view_control()
        ctr.rotate(180, 90.0)

    # In live mode, update the visualizer to show the updated point cloud
    def update(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.reset_view_point(True)
        ctr = self.vis.get_view_control()
        ctr.rotate(0, 500)
        self.vis.update_renderer()

    # DEPRECATED
    # Returns the rotation matrix for a rotation in the Y axis
    def rotateY(self, d):
        return np.array([[np.cos(d), 0, np.sin(d)],
                         [0, 1, 0], 
                         [-np.sin(d), 0, np.cos(d)]])

    # DEPRECATED
    # Returns the rotation matrix for a rotation in the X axis
    def rotateX(self, d):
        return np.array([[1, 0, 0],
                         [0, np.cos(d), -np.sin(d)], 
                         [0, np.sin(d), np.cos(d)]])

    # Move a single frame's point cloud to match it's location in the global map
    def movePoints(self, pcd, transformID):
        # vecotor magnitidue from addition of x and y component
        # TODO: CURRENTLY USING CONSTANT VALUE
        if True or transformID[2] is None or transformID[3] is None:
            magnitude = 0.00005
        else:
            magnitude = np.hypot(transformID[2], transformID[3]) / 100000

        # Forward
        if transformID[0] == 'f':
            # Forward movement is mapped to the X-Z plane based on the vector angle of the current rotation
            #   ^                       |
            #   |                       |   
            #   |       <------         |           ------->
            #   | 0d            90d     v 180d               270d
            self.xyz += np.array([np.sin(self.rot[1][0])*magnitude, 0, -np.cos(self.rot[1][0])*magnitude])
        
        # Rotate right (Y axis)
        if transformID[1] == 'r':
            self.rot += np.array([0,0.032,0]).reshape(3,1)
            self.xyz += np.array([np.sin(self.rot[1][0])*magnitude, 0, -np.cos(self.rot[1][0])*magnitude])
        
        # Rotate left (Y axis)
        elif transformID[1] == 'l':
            self.rot += np.array([0,-0.032,0]).reshape(3,1)
            self.xyz += np.array([np.sin(self.rot[1][0])*magnitude, 0, -np.cos(self.rot[1][0])*magnitude])
        
        # Rotate up (X axis)
        elif transformID[1] == 'u':
            self.rot += np.array([-0.032,0,0]).reshape(3,1)
        
        # Rotate down (X axis)
        elif transformID[1] == 'd':
            self.rot += np.array([0.032,0,0]).reshape(3,1)

        # Apply transformation
        cur_pcd = pcd.translate(self.xyz)
        cur_pcd = pcd.rotate(self.rot)
        return cur_pcd

    # Add the point cloud from the current frame to the global point cloud of the map
    def estimate(self, img_colour, img_depth, transformID, crop_fact_h=0.8, crop_fact_w=0.7, downsample=20):
        img_colour = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)

        # Crop the frame to reduce boundary depth noise
        h, w = img_colour.shape[:2]
        crop_h = int((h - (crop_fact_h*h)) / 2)
        crop_w = int((w - (crop_fact_w*w)) / 2)
        
        # Convert the cv2 frames to the Open3D image format
        img_colour = copy.deepcopy(img_colour[crop_h:h-crop_h, crop_w:w-crop_w, :])
        img_od3_colour = o3d.geometry.Image(img_colour)
        img_depth = copy.deepcopy(img_depth[crop_h:h-crop_h, crop_w:w-crop_w, :])
        img_od3_depth = o3d.geometry.Image(img_depth)
        
        # Create a point cloud from the current frame and transform it so it is right side up
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img_od3_colour, img_od3_depth, convert_rgb_to_intensity=False)
        cur_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        cur_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Downsample the number of points to reduce output size and rendering computation
        cur_pcd = cur_pcd.uniform_down_sample(downsample)
        
        # Cut a square in the point cloud to make a path viewable when the point cloud is rendered
        #   _______________
        #   |   X X X X   |
        #   |   X X X X   |
        #   |   X X X X   |
        #   |             |  
        #   --------------- where X represensts the cut region
        pcd_array = np.asarray(cur_pcd.points)
        pcd_cent = cur_pcd.get_center()
        pcd_max = cur_pcd.get_max_bound()
        pcd_min = cur_pcd.get_min_bound()
        x_thresh = (pcd_max[0] - pcd_cent[0]) / 2
        y_thresh = (pcd_max[1] - pcd_cent[1]) / 2
        xxR = np.where(pcd_array[:,0] > pcd_cent[0] + x_thresh)
        xxL = np.where(pcd_array[:,0] < pcd_cent[0] - x_thresh)
        yy = np.where(pcd_array[:,1] < pcd_cent[1] - y_thresh)
        uu = np.unique(np.append(np.append(xxR[0], xxL[0]), yy[0]))
        cur_pcd = cur_pcd.select_down_sample(uu, invert=False)

        # Based on camera movement, adjust the placement of the point cloud in the global map
        cur_pcd = self.movePoints(cur_pcd, transformID)
        
        # Add the point cloud to the global map
        if self.pcd == None:
            self.pcd = copy.deepcopy(cur_pcd)
            if self.mode == 'online':
                self.vis.add_geometry(self.pcd)
        else:
            self.pcd += cur_pcd

        # Render the current global map
        if self.mode == 'online':
            self.update()
        
        outFramePCD = cv2.cvtColor(cv2.normalize(np.asarray(
                                                    self.vis.capture_screen_float_buffer(True)),
                                                    None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLOR_RGB2BGR)

        return [self.pcd, outFramePCD]