from scipy import spatial
import time
import numpy as np
from result_set import KNNResultSet
import struct

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

def scipyKdtreeSearch(tree:spatial.KDTree,result_set:KNNResultSet,point: np.ndarray):
    scipy_nn_dis,scipy_nn_idx=tree.query(point,result_set.capacity)
    for idx, distindex in enumerate(result_set.dist_index_list):
        distindex.distance=scipy_nn_dis[idx]
        distindex.index=scipy_nn_idx[idx]
    return False

def main():
    filename = "/Users/zachary/Desktop/3D Point Clouds/2 Nearest Neighbor Problem/lesson2code/data/000000.bin"
    db_np = read_velodyne_bin(filename)
    k=8
    construction_time_sum = 0
    knn_time_sum = 0

    begin_t = time.time()
    tree = spatial.KDTree(db_np)
    construction_time_sum += time.time() - begin_t

    for index ,p in enumerate(db_np):
        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        scipyKdtreeSearch(tree,result_set,p)
        knn_time_sum += time.time() - begin_t
    
    construction_time_sum *= 1000
    knn_time_sum *= 1000
    
    print('scipy kdtree building: %.5f' % construction_time_sum)
    print('scipy kdtree: %.5f' % knn_time_sum)

if __name__ == '__main__':
    main()