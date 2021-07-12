# Implement voxel filtering

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# function：apply voxel filtering on point cloud
# input：
#     point_cloud
#     leaf_size: voxel size
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []

    # hw3
    # start the code

    # step1: compute the min or max of the point
    x_max, y_max, z_max = point_cloud.max(axis=0)
    x_min, y_min, z_min = point_cloud.min(axis=0)

    # step2: determine the voxel grid size r
    voxel_grid_size = leaf_size

    # step3: Compute the dimension of the voxel grid
    Dx = (x_max - x_min) / voxel_grid_size
    Dy = (y_max - y_min) / voxel_grid_size
    Dz = (z_max - z_min) / voxel_grid_size

    # step4: Compute voxel index for each point
    point_cloud = np.asarray(point_cloud)
    h = []
    for i in range(point_cloud.shape[0]):
        hx = np.floor((point_cloud[i][0] - x_min) / voxel_grid_size)
        hy = np.floor((point_cloud[i][1] - x_min) / voxel_grid_size)
        hz = np.floor((point_cloud[i][2] - x_min) / voxel_grid_size)
        H = hx + hy * Dx + hz * Dx * Dy
        h.append(H)
    h = np.asarray(h)

    # step5: Sort the points according to the index in step4
    voxel_index = np.argsort(h)
    h_sort = h[voxel_index]

    # step6: Iterate the sorted points, select points according to Centroid / Random method
    index_begin = 0
    for i in range(len(voxel_index) - 1):
        if (h_sort[i] == h_sort[i + 1]):
            continue

        point_index = voxel_index[index_begin:(i + 1)]
        filtered_points.append(np.mean(point_cloud[point_index], axis=0))
        index_begin = i

    # end the code

    # change point cloud format to np.array
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    """ ********************* load point cloud data ************************** """
    # instance number, range: 0-39, 40 instances
    category_index = 3
    # path for point cloud dataset
    root_dir = 'D:\PointCloud\modelnet40_normal_resampled'
    category = os.listdir(root_dir)
    point_cloud_name = '_0005.txt'
    # default first point cloud data
    file_name = os.path.join(root_dir,
                             category[category_index],
                             category[category_index] +
                             point_cloud_name)
    print(file_name)
    # six columns, the first three values are coordinates, the latter three are normals.
    point_cloud_np = np.loadtxt(file_name, delimiter=',')
    # get points from point cloud, we only process coordinates and ignore the normals
    point_cloud_np = point_cloud_np[:, 0:3]
    print('total points number is:', point_cloud_np.shape[0])

    """ *************** raw point cloud visualization ************************** """
    point_cloud_pd = pd.DataFrame(point_cloud_np)
    point_cloud_pd.columns = ["x", "y", "z"]
    point_cloud_pynt = PyntCloud(point_cloud_pd)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    """ *************** apply voxel filtering ************************** """
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.3)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # visualize point cloud after filtering
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
