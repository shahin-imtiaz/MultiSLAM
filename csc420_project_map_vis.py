import argparse
import open3d as o3d

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True, help="Point cloud file path")
    args = vars(ap.parse_args())

    pcd = o3d.io.read_point_cloud(args['input'])
    o3d.visualization.draw_geometries([pcd])