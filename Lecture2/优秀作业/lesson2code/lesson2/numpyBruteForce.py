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

def bruteSearch(db: np.ndarray,result_set:KNNResultSet, query: np.ndarray):
    diff = np.linalg.norm(np.expand_dims(query, 0) - db, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]

def main():
    filename = "/Users/zachary/Desktop/3D Point Clouds/2 Nearest Neighbor Problem/lesson2code/data/000000.bin"
    db_np = read_velodyne_bin(filename)
    k=8
    brute_time_sum=0
    query = db_np[0,:]
    
    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    bruteSearch(db_np,result_set,query)
    brute_time_sum += time.time() - begin_t
    print('brute:',brute_time_sum*1000)

if __name__ == '__main__':
    main()