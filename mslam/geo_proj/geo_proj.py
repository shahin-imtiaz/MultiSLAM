import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import copy
# from open3d import geometry

class GeoProjection():
    def __init__(self, mode='offline'):
        self.mode = mode
        self.pcd = None
        self.z = 0
        self.vis = None

        if self.mode == 'online':
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

    # def init_geometry(self, img):
    #     self.vis.add_geometry(pcd2)
    
    def update(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def estimate(self, img_colour, img_depth, crop_fact_h=0.8, crop_fact_w=0.7, downsample=20):
        # crop
        h, w = img_colour.shape[:2]
        crop_h = int((h - (crop_fact_h*h)) / 2)
        crop_w = int((w - (crop_fact_w*w)) / 2)
        
        # print(h, w)

        img_colour = copy.deepcopy(img_colour[crop_h:h-crop_h, crop_w:w-crop_w, :])
        img_od3_colour = o3d.geometry.Image(img_colour)
        img_depth = copy.deepcopy(img_depth[crop_h:h-crop_h, crop_w:w-crop_w, :])
        img_od3_depth = o3d.geometry.Image(img_depth)

        # o3d.visualization.draw_geometries([img_od3_colour])
        # o3d.visualization.draw_geometries([img_od3_depth])
        
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img_od3_colour, img_od3_depth)

        cur_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        cur_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cur_pcd = cur_pcd.uniform_down_sample(downsample)
        cur_pcd.translate([0,0,self.z])
        self.z -= 0.00001
        
        if self.pcd == None:
            self.pcd = copy.deepcopy(cur_pcd)
            
            if self.mode == 'online':
                self.vis.add_geometry(self.pcd)
        else:
            # self.pcd.points = cur_pcd.points
            # self.pcd.colors = cur_pcd.colors
            # self.pcd.normals = cur_pcd.normals
            self.pcd += cur_pcd
            # self.pcd = copy.deepcopy(cur_pcd)
            # self.vis.add_geometry(self.pcd)
        
        if self.mode == 'online':
            self.update()
            return None
        else:
            return self.pcd
        
    #     if self.pcd == None:
    #         self.pcd = 
    # # o3d.visualization.draw_geometries([pcd2])
    # # pts = np.mgrid[1: 6: complex(100),
    # #       2: 9: complex(100),
    # #       3: 6: complex(100)].reshape(3, -1).T
    # # pcd = open3d.geometry.PointCloud()

    # # pcd.points = open3d.utility.Vector3dVector(pts)
    # pcd.points = pcd2.points
    # vis.add_geometry(pcd2)
    # # vis.update_geometry()
    # # vis.poll_events()
    # # vis.update_renderer()
    # # time.sleep(20)


if __name__ == "__main__":
    img_colour = cv2.cvtColor(cv2.imread("pc_test1.jpg"), cv2.COLOR_BGR2RGB)
    img_depth = cv2.imread("pc_test1_d.jpg")

    img_colour = o3d.geometry.Image(img_colour)
    img_depth = o3d.geometry.Image(img_depth)

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img_colour, img_depth, convert_rgb_to_intensity=False)

    pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pc.remove_radius_outlier(10, 10)
    # print(pc.get_max_bound(), pc.get_min_bound())
    # pc.points = o3d.utility.Vector3dVector(np.random.randint(-3,4,3000).reshape(1000,3))
    # print(pc.get_center())
    pc_array = np.asarray(pc.points)
    # print("max", pc.get_max_bound())
    # print("min", pc.get_min_bound())
    
    pc_cent = pc.get_center()
    pc_max = pc.get_max_bound()
    pc_min = pc.get_min_bound()
    x_thresh = (pc_max[0] - pc_cent[0]) / 2
    y_thresh = (pc_max[1] - pc_cent[1]) / 2

    xxR = np.where(pc_array[:,0] > pc_cent[0] + x_thresh)
    xxL = np.where(pc_array[:,0] < pc_cent[0] - x_thresh)
    yy = np.where(pc_array[:,1] < pc_cent[1] - y_thresh)

    uu = np.unique(np.append(np.append(xxR[0], xxL[0]), yy[0]))

    pc_redo = pc.select_down_sample(uu, invert=False)
    print('uu', uu.shape)
    print(pc.points)
    print(pc_redo.points)
    # print(o3d.geometry.PointCloud.get_rotation_matrix_from_xyz)
    # print(o3d.geometry.XYZ((0.75, 0.5, 0)))
    # pc_redo.rotate(o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz((0.75, 0.5, 0)))
    # print(pc_redo.get_rotation_matrix_from_axis_angle(np.array([0,0,1])))

    # pc = pc.rotate(np.array([0,3.141595,0]).reshape(3,1))
    # o3d.visualization.draw_geometries([pc])
    # pc.estimate(img_colour, img_depth, downsample=1)
    # time.sleep(15)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)
    vis.update_geometry()
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.rotate(0, 1000)
    vis.poll_events()
    vis.update_renderer()
    img_vis = np.asarray(vis.capture_screen_float_buffer(False))
    img_vis = cv2.cvtColor(cv2.normalize(img_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLOR_RGB2BGR)
    cv2.imwrite("mmm.png", img_vis)













    # # (flag1, img_colour_encodedImage) = cv2.imencode(".jpg", img_colour)
    # # (flag2, img_depth_encodedImage) = cv2.imencode(".jpg", img_depth)
    # # print(flag1, flag2)
    # # img_colour_encodedImage = (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
    # #         bytearray(img_colour_encodedImage) + b'\r\n')
    # # img_depth_encodedImage = (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
    # #         bytearray(img_depth_encodedImage) + b'\r\n')
    # # img_colour = o3d.io.read_image("pc_test1.jpg")
    # # img_depth = o3d.io.read_image("pc_test1_d.jpg")
    # # print(np.asarray(img_colour).shape)
    # # exit(1)
    # rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     img_colour, img_depth)
    # # exit(1)
    # # plt.subplot(1, 2, 1)
    # # plt.title('Redwood grayscale image')
    # # plt.imshow(rgbd_img.color)
    # # plt.subplot(1, 2, 2)
    # # plt.title('Redwood depth image')
    # # plt.imshow(rgbd_img.depth)

    # pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry(pcd)
    # # opt = vis.get_render_option()
    # # opt.point_size = 1.0

    # pcd1 = o3d.geometry.PointCloud()

    # pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_img,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # # o3d.visualization.draw_geometries([pcd2])
    # # pts = np.mgrid[1: 6: complex(100),
    # #       2: 9: complex(100),
    # #       3: 6: complex(100)].reshape(3, -1).T
    # # pcd = open3d.geometry.PointCloud()

    # # pcd.points = open3d.utility.Vector3dVector(pts)
    # pcd.points = pcd2.points
    # vis.add_geometry(pcd2)
    # # vis.update_geometry()
    # # vis.poll_events()
    # # vis.update_renderer()
    # # time.sleep(20)
    # while True:

    # #     # pts = np.mgrid[1: np.random.randint(10): complex(100),
    # #     #         2: np.random.randint(10): complex(100),
    # #     #         3: np.random.randint(10): complex(100)].reshape(3, -1).T
        
    # #     # pcd.points = pcd1.points
    # #     # # pcd.clear()
    # #     # # pcd.points = open3d.utility.Vector3dVector(pts)
    #     vis.update_geometry()
    #     vis.poll_events()
    #     vis.update_renderer()
    #     # time.sleep(5)
    # #     x = o3d.utility.Vector3dVector([[1, 2, 3]])
    # #     pcd.points = pcd.translate(x)
    # #     # pcd.clear()
    # #     # pcd.points = open3d.utility.Vector3dVector(pts)
    # #     vis.update_geometry()
    # #     vis.poll_events()
    # #     vis.update_renderer()
    #     # time.sleep(0.05)