"""
    Function:
    1. Load point cloud data from dataset
    2. filter ground point cloud from dataset
    3. Cluster over the remaining dataset

"""

import numpy as np
import math
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

import open3d as o3d
import voxel_filter as filter
from sklearn.neighbors import KDTree
import time


"""
    Function: Read point cloud from KITTI dataset.
    
    Input:
            path: .bin files directory
    return:
            homography matrix of the point cloud (numpy array, N*3)
"""
def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# -------------------------------------------- Ground Filtering --------------------------------------------
"""
    Function: Normal Estimation. Three points in the plane are known. 
              Compute the plane equation and return the coefficient.
            
    Input:
            points: data points, 3*3 array
            normalize: bool
    return:
            coefficient: 1*4 array
"""
def estimate_normal(points, normalize=True):
    """
        plane: ax + by + cz + d = 0
        Normal:
            a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
            b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
            c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            d = -a * x1 - b * y1 - c * z1
    """
    p21 = points[1, :] - points[0, :]
    p31 = points[2, :] - points[0, :]
    a = (p21[1] * p31[2]) - (p31[1] * p21[2])
    b = (p21[2] * p31[0]) - (p31[2] * p21[0])
    c = (p21[0] * p31[1]) - (p31[0] * p21[1])

    if normalize:
        r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        a /= r
        b /= r
        c /= r

    d = -a * points[0, 0] - b * points[0, 1] - c * points[0, 2]

    return np.array([a, b, c, d])



"""
    Function: Remove ground from point cloud dataset
    
    Input:
            data: point cloud, one frame, N*3
            sample: the number of randomly selected samples
            max_iter: maximum iterations
            dist_thre: distance threshold, used to determine whether a data point belongs to inlier
            prob: the probability of non-ground data points over all the data points
            
    return:
            non_ground_point_idx: indices of non-ground data points
            ground_point_idx: indices of ground data points
            
"""
def ground_segmentation(data, sample=3, max_iter=100, dist_thre=0.35, prob=0.75):
    N, D = data.shape # N: the number of sample, D: dimension
    num_ground = 0 # the number of ground data points
    max_num_ground = 0 # the maximum number of ground data points
    non_ground_points = []
    ground_points = []

    """ ******************** Using RANSAC to detect the ground  ******************** """
    for i in range(max_iter) :
        # ----------------- Step 1: Randomly Select Samples (Plane detection: 3 samples selected) -----------------
        selected_samples = random.sample(data.tolist(), sample)
        selected_samples = np.array(selected_samples)

        # ----------------- Step 2: Get Plane Coefficients -----------------
        normal = estimate_normal(selected_samples, normalize=True).reshape(-1, 1)

        # ----------------- Step 3: Compute the distances of all the points to the plane -----------------
        distances = np.abs(np.dot(data, normal[:3]) + normal[3])

        # ----------------- Step 4: Compute the number of points in this plane -----------------
        points_idx_in_plane = [idx for idx in range(len(distances)) if distances[idx] < dist_thre]
        num_ground = len(points_idx_in_plane)  # update the number of ground points[i] < dist_thre]

        # ----------------- Step 5: Select the plane with most inlier points -----------------
        if num_ground > max_num_ground:
            max_num_ground = num_ground
            ground_points = points_idx_in_plane
            non_ground_points = [idx for idx in range(len(distances)) if distances[idx] >= dist_thre]

        # ----------------- Step 6: Stop iteration if the number of ground points reaches the prob -----------------
        if (max_num_ground / N) > (1 - prob):
            break

    print('origin data points number:', N)
    print('segmented data points number (non-ground):', len(non_ground_points))
    return non_ground_points, ground_points


# -------------------------------------------- DBSCAN (Object Clustering) --------------------------------------------
"""
    Function: Search for core points from point clouds. 
              Constraints: (1) in the range of radius; (2) the number of points >= min_num_samples
              Euclidean space can be used to search for the points. kdtree can also be used for faster search.
              
    Input: 
            data: point clouds
            radius: search range
            min_num_samples: the minimum number of samples
            method: searching strategy
    Return:
            core_points(set): set of core point indices
            neighbor_points(dict): nearest neighbor points around the core point (include core point)
"""
def getCore(data, radius, min_num_samples, method='kdtree'):
    """
        Function: Get the neighbor points of a specific point in the range of radius based on the dataset
        Input:
                point: target point, 1*3
                data: point clouds, N*3
                radius: search range
        Return:
                point_to_neighbors: indices of neighbor points
    """
    def getNeighbors(point, data, radius):
        distances = np.sum((point - data) ** 2, axis=1) # distances square of all points to the center point
        neighbor_points = [idx for idx in range(distances.shape[0]) if distances[idx] < radius * radius]
        return neighbor_points

    core_points = set()
    point_to_neighbors = {}

    if method == 'euclidean':
        for i in range(data.shape[0]):
            neighbors = getNeighbors(data[i], data, radius)
            if len(neighbors) >= min_num_samples:
                core_points.add(i)
                point_to_neighbors[i] = neighbors

    if method == 'kdtree':
        kdtree = KDTree(data, leaf_size=1) # leaf_size can be adjusted
        neighbors = kdtree.query_radius(data, radius) # indices of nearest neighbors in the range of radius
        for i in range(neighbors.shape[0]):
            if len(neighbors[i]) >= min_num_samples:
                core_points.add(i)
                point_to_neighbors[i] = neighbors[i]

    return core_points, point_to_neighbors



"""
    Function: Cluster the point clouds using DBSCAN.
    Input:
            data: point clouds (non-ground points removed)
            radius: search range
            min_num_samples: the minimum number of samples around the core points
    return:
            clusters_index: cluster labels of all non-ground points, N*1, N does not include ground points.

"""
def clustering(data, radius=0.4, min_num_samples=5):
    N, _ = data.shape

    # --------------------- Step 1: Initialization ---------------------
    core_points = set()
    point_to_neighbors = {}
    cluster = set()
    clusters_index = np.zeros(N, dtype=int)
    # noise = []
    points_not_visit = set(range(N))
    k = 0 # the kth cluster

    # --------------------- Step 2: Search for all the core points ---------------------
    core_points, point_to_neighbors = getCore(data, radius, min_num_samples, method='kdtree')

    # traverse all the core points
    while len(core_points):
        points_old = points_not_visit
        idx = np.random.randint(0, len(core_points)) # randomly select a core point index
        core_point = list(core_points)[idx]
        points_not_visit = points_not_visit - {core_point} # delete current core point from unvisited points set

        # --------------------- Step 3: BFS ---------------------
        queue = []
        queue.append(core_point)
        while len(queue):
            cur_point = queue[0]
            if cur_point in core_points:
                neighbors = set(point_to_neighbors[cur_point]) # find the neighbors of current core point
                neighbors = neighbors & points_not_visit # exclude the core point in neighbors set
                queue += list(neighbors) # add the neighbors to the queue
                points_not_visit = points_not_visit - neighbors # mark the neighbors as visited points
            queue.remove(cur_point)

        # --------------------- Step 4: Get clusters ---------------------
        cluster = points_old - points_not_visit
        core_points = core_points - cluster
        k += 1
        clusters_index[list(cluster)] = k

    return clusters_index



"""
    Function: display cluster results using open3d
"""
def plot_clusters_o3d(data, cluster_index):
    def map(color_index, num_clusters):
        color = [0]*3
        color = [color_index/num_clusters] *3
        return color

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.5,0.5,0.5])
    num_clusters = cluster_index.max() + 1
    pcd.colors = o3d.utility.Vector3dVector([
        map(label, num_clusters) for label in cluster_index
    ])
    # visualize
    o3d.visualization.draw_geometries([pcd])



"""
    Function: Display cluster point clouds. Paint each cluster with different colors.
    Input: 
            data: point clouds (non-ground data)
            cluster_index: cluster labels
"""
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()



"""
    Function: Display point clouds and corresponding indices
    Input:
            data: point cloud, N*3
            index: point cloud indices
"""
def plot_pt(data, index = None):
    pcd = o3d.geometry.PointCloud()
    if index != None:
        pcd.points = o3d.utility.Vector3dVector(data[index])
    else:
        pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])



def main():
    root_dir = 'data/'
    category = os.listdir(root_dir)
    category = category[:]
    iteration_num = len(category)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, category[i])
        print('clustering point cloud file:', filename)

        print('------------- Display Original Point Cloud ----------------')
        origin_points = read_velodyne_bin(filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        o3d.visualization.draw_geometries([pcd])


        print('---------------- Filter Ground Point Cloud -------------------')
        segmented_points_idx, ground_point_idx = ground_segmentation(data=origin_points,
                                                                     dist_thre=0.25,
                                                                     prob=0.4) # Filtering Method：RANSAC
        plot_pt(origin_points, ground_point_idx) # visualize ground point cloud
        plot_pt(origin_points, segmented_points_idx) # visualize non-ground point cloud


        print('------------- Optimization: sampling ------------')
        # perform sampling on original non-ground point cloud, reduce the data size
        segmented_points = origin_points[segmented_points_idx]
        filter_points = filter.voxel_filter(segmented_points, leaf_size=0.1)
        print('Data points number before sampling: ', len(segmented_points_idx))
        print('Data points number after sampling: ', filter_points.shape[0])
        plot_pt(segmented_points)  # visualize non-ground point cloud before sampling
        plot_pt(filter_points)  # visualize non-ground point cloud after sampling


        print('------------ Cluster over non-ground point cloud -----------')
        # cluster over original non-ground point cloud
        start_t = time.time()
        cluster_index = clustering(segmented_points)
        origin_t = time.time() - start_t

        # cluster over sampled non-ground point cloud
        start_t = time.time()
        filter_cluster_index = clustering(filter_points)
        filter_t = time.time() - start_t

        print('Original non-ground point cloud clustering time', origin_t)
        print('Sampled non-ground point cloud clustering time：', filter_t)

        # plot_clusters_o3d(segmented_points, cluster_index)
        plot_clusters(segmented_points, cluster_index)
        # plot_clusters_o3d(filter_points, filter_cluster_index)
        plot_clusters(filter_points, filter_cluster_index)


if __name__ == '__main__':
    main()
