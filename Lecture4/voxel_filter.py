# Voxel filtering

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

"""
    Function: Read point cloud file
    Input:
            filename: file directory
            separator: delimit
    Return:
            point(np.array): data points, Nx6
"""
def readXYZfile(filename, Separator = " "):
    data = [[],[],[],[],[],[]]
    
    num = 0
    for line in open(filename, 'r'): # read point cloud line by line
        line = line.strip('\n') # remove line break
        a,b,c,d,e,f = line.split(Separator)
        data[0].append(a) # X coordinate
        data[1].append(b) # Y coordinate
        data[2].append(c) # Z coordinate
        data[3].append(d)
        data[4].append(e)
        data[5].append(f)
        num = num + 1

    #string to float
    x = [float(data[0]) for data[0] in data[0]]
    y = [float(data[1]) for data[1] in data[1]]
    z = [float(data[2]) for data[2] in data[2]]
    nx = [float(data[3]) for data[3] in data[3]]
    ny = [float(data[4]) for data[4] in data[4]]
    nz = [float(data[5]) for data[5] in data[5]]
    print("The number of points is: {}".format(num))
    point = [x, y, z, nx, ny, nz]
    point = np.array(point) # list to np.array 

    point = point.transpose() # 6*N to N*6
    return point


"""
    Function: Apply Voxel filtering on point cloud
    Input: 
            point_cloud: input data
            leaf_size: voxel size
            method: downsample method, centroid or random, default: centroid
"""
def voxel_filter(point_cloud, leaf_size, method = 'centroid'):
    filtered_points = []

    # get the range of bounding box of point cloud
    x_max = np.max(point_cloud[:,0], axis = 0)
    x_min = np.min(point_cloud[:,0], axis =0)
    y_max = np.max(point_cloud[:,1], axis =0)
    y_min = np.min(point_cloud[:,1], axis = 0)
    z_max = np.max(point_cloud[:,2], axis =0)
    z_min = np.min(point_cloud[:,2], axis =0)

    # Compute the dimension of the voxel grid
    Dx = ((x_max - x_min)/leaf_size).astype(np.int)
    Dy = ((y_max - y_min)/leaf_size).astype(np.int)
    Dz = ((z_max - z_min)/leaf_size).astype(np.int)

    # Compute voxel index for each point
    hx = ((point_cloud[:,0]- x_min)/leaf_size).astype(np.int)
    hy = ((point_cloud[:,1]- y_min)/leaf_size).astype(np.int)
    hz = ((point_cloud[:,2]- z_min)/leaf_size).astype(np.int)
    idx = np.dtype(np.int64)
    idx= hx + hy * Dx + hz * Dx * Dy # get the index of each point

    point_cloud_idx = np.insert(point_cloud, 0, values = idx, axis = 1) # combine index and data points
    #point_cloud_idx = np.c_[idx, point_cloud]

    # Sort by the index
    point_cloud_idx = point_cloud_idx[np.lexsort(point_cloud_idx[:,::-1].T)]
    #print(point_cloud_idx[0:15,:])

    # Select points according to centroid/random method
    point_cloud_idx[:,0].astype(np.int)
    n = 0
    k = point_cloud_idx[0,0]
    if method == 'centroid':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                # print(np.mean(point_cloud_idx[n:i, :], axis = 0))
                filtered_points.append(np.mean(point_cloud_idx[n:i,1:4], axis =0))
                k = point_cloud_idx[i,0]
                n = i
    elif method == 'random':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                # print(np.mean(point_cloud_idx[n:i, :], axis = 0))
                point_rand = np.random.randint(n,i) # randomly select a point in the range of [n, i)
                filtered_points.append(point_cloud_idx[point_rand,1:4])
                k = point_cloud_idx[i,0]
                n = i



    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # load point cloud from ModelNet dataset
    # cat_index = 10 # catogry number，0-39
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # path
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply')
    # point_cloud_pynt = PyntCloud.from_file(file_name)


    #abs_path = os.path.abspath(os.path.dirname(__file__))
    #filename = os.path.join(abs_path, 'car_0001.txt')

    root_dir = '/home/magictz/Projects/shenlan/dataset/modelnet40_normal_resampled'
    filenames = os.path.join(root_dir, 'modelnet40_shape_names.txt')
    filename = []


    for line in open(filenames, 'r'):
        line = line.strip('\n')
        filename =  os.path.join(root_dir, line, line+'_0001.txt')

        pointcloud = readXYZfile(filename, Separator= ',') 

        # method 1: use pyntcloud to read .txt and create point cloud object
        # pointcloud_pynt = PyntCloud.from_file(filename, 
        #                                                         sep=",", 
        #                                                         header =-1, 
        #                                                         names = ["x", "y", "z"])
        # pcd = pointcloud_pynt.to_instance("open3d", mesh = False)
        # Method 2: use customized readXYZfile()
        pcd = o3d.geometry.PointCloud() # create point cloud object
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:,0:3]) # read x,y,z
        pcd2 = o3d.geometry.PointCloud()
        o3d.visualization.draw_geometries([pcd])
        
        # call voxel filtering function
        filtered_cloud = voxel_filter(pointcloud[:,0:3], 0.07, method= 'random') # voxel grid resolution: 0.1m
        filtered_cloud_c = voxel_filter(pointcloud[:,0:3], 0.07, method= 'centroid') # voxel grid resolution: 0.1m
        pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
        pcd2.points = o3d.utility.Vector3dVector(filtered_cloud_c)


        # display the point cloud after filtering
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([pcd2])



if __name__ == '__main__':
    main()
