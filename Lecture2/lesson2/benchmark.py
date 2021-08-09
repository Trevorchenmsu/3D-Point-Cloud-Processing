# construct trees (kdtree and octree) and query based on the point cloud data in the dataset
# Test the running time in kdtree and octree separately

import random
import math
import numpy as np
import time
import os
import struct

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet
from scipy import spatial

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32).T

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1
    num_queries = 1 # the number of query points

    # root_dir = '/Users/renqian/cloud_lesson/kitti' # dataset directory
    # cat = os.listdir(root_dir)
    # iteration_num = len(cat)

    """ ***************** file path definition and data reading ********************"""
    file_dir = os.path.abspath(os.path.dirname(__file__)) # dataset stored in current path
    file_name = os.path.join(file_dir, '000000.bin')
    iteration_num = 1
    db_np_raw = read_velodyne_bin(file_name).T
    db_np_idx = np.random.choice(db_np_raw.shape[0], size=(10,))  # randomly sampling 10 points

    """ *********************** Octree Testing *********************** """

    print("-------------------- Octree Testing --------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        # filename = os.path.join(root_dir, cat[i])
        # db_np = read_velodyne_bin(filename)

        # -------------------- Construct Octree --------------------
        begin_t = time.time()
        root = octree.octree_construction(db_np_raw, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        print ("The number of query points is: ", db_np_idx.shape[0])

        # -------------------- Search for query points --------------------
        for idx in db_np_idx:
            query = db_np_raw[idx,:]
            # query = db_np[0,:]

            # ------------ KNN search --------------
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            octree.octree_knn_search(root, db_np_raw, result_set, query)
            knn_time_sum += time.time() - begin_t

            # ------------ Radius search --------------
            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            # octree.octree_radius_search(root, db_np, result_set, query)
            octree.octree_radius_search_fast(root, db_np_raw, result_set, query)
            radius_time_sum += time.time() - begin_t



            # ------------ Brute force search --------------
            begin_t = time.time()
            diff = np.linalg.norm(np.expand_dims(query, 0) - db_np_raw, axis=1)
            nn_idx = np.argsort(diff)
            nn_dist = diff[nn_idx]
            brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000/iteration_num))

    """ *********************** Kdtree Testing *********************** """
    print("-------------------- Kdtree Testing --------------------")

    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    knn_scipy_time_sum = 0
    radius_scipy_time_sum = 0

    for i in range(iteration_num):
        # filename = os.path.join(root_dir, cat[i])
        # db_np = read_velodyne_bin(filename)

        # ----------------- Construct Kdtree -----------------
        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np_raw, leaf_size)
        construction_time_sum += time.time() - begin_t
        kdtree_scipy = spatial.KDTree(db_np_raw) # use scipy library to construct kdtree


        print("The number of query points is: ", db_np_idx.shape[0])

        # ----------------- Search for query points -----------------
        for idx in db_np_idx:
            query = db_np_raw[idx,:]
            # query = db_np[0,:]

            # ------------ KNN search --------------
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            kdtree.kdtree_knn_search(root, db_np_raw, result_set, query)
            knn_time_sum += time.time() - begin_t

            # ------------ radius search --------------
            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            kdtree.kdtree_radius_search(root, db_np_raw, result_set, query)
            radius_time_sum += time.time() - begin_t

            # ------------ Brute force search --------------
            begin_t = time.time()
            diff = np.linalg.norm(np.expand_dims(query, 0) - db_np_raw, axis=1)
            nn_idx = np.argsort(diff)
            nn_dist = diff[nn_idx]
            brute_time_sum += time.time() - begin_t

            # ------------ KNN search using scipy library --------------
            begin_t = time.time()
            kdtree_scipy.query(query, k=k)
            knn_scipy_time_sum += time.time() - begin_t

            # ------------ Radius search using scipy library --------------
            begin_t = time.time()
            kdtree_scipy.query_ball_point(query, radius)
            radius_scipy_time_sum += time.time() - begin_t

    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f, knn_scipy %.3f, radius_scipy %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num,
                                                                     knn_scipy_time_sum * 1000 / iteration_num,
                                                                     radius_scipy_time_sum * 1000 / iteration_num,
                                                                     ))



if __name__ == '__main__':
    main()