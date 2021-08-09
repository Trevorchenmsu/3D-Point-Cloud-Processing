# Implement PCA and normal estimation

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# test open3d library
# def main():
#     np_pc = np.random.random((1000,3))
#     pc_view = o3d.geometry.PointCloud()
#     pc_view.points = o3d.utility.Vector3dVector(np_pc)
#     o3d.visualization.draw_geometries([pc_view])
#
# if __name__ == "__main__":
#     main()


"""
    method：compute PCA
    input：
             data：point cloud，N X 3 matrix
             correlation：distinguish np.cov and np.corrcoef，fault=False
             sort: eigenvalues sorting. default=True
    return：
             eigenvalues
             eigenvectors
"""

def PCA(data, correlation=False, sort=True):
    # Start the code

    # 1. normalize the data to be zero mean
    data_mean = np.mean(data, axis=0)
    data_normalized = data - data_mean

    # 2. get covariance matrix
    func = np.cov if not correlation else np.corrcoef
    cov_matrix = func(data_normalized, rowvar=False, bias=True)

    # 3. singular value decomposition
    eigenvectors, eigenvalues, eigenvectors_transpose = np.linalg.svd(cov_matrix, full_matrices=True)

    # End the code

    if sort:
        # argsort() is increasing sorting. with -1,
        # it becomes decreasing sorting.
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    """ ********************* load point cloud data ************************** """
    # instance number, range: 0-39, 40 instances
    category_index = 5
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
    o3d.visualization.draw_geometries([point_cloud_o3d])

    """ *************** apply PCA to get principal directions ************************** """
    points = np.asarray(point_cloud_o3d.points)
    eigenvalues, eigenvectors = PCA(points)
    point_cloud_vector1 = eigenvectors[:, 0]
    point_cloud_vector2 = eigenvectors[:, 1]
    point_cloud_vector3 = eigenvectors[:, 2]
    print('the first component of this point cloud is: ', point_cloud_vector1)
    print('the second component of this point cloud is: ', point_cloud_vector2)
    print('the third component of this point cloud is: ', point_cloud_vector3)

    """ *************** visualize the principal components in the point cloud ************************** """
    point = [[0, 0, 0], point_cloud_vector1, point_cloud_vector2, point_cloud_vector3]
    lines = [[0, 1], [0, 2],[0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

    """ *************** compute the normal of each point iteratively ************************** """
    # store raw point cloud data to KD tree
    # prepare for applying nearest neighbor method to get points
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []

    # start the code
    for i in range(points.shape[0]):
        """
        search_knn_vector_3d function
        input: [each point, the number of KNN]
        return: [int, open3d.utility.IntVector, open3d.utility.DoubleVector]

        find 10 KNN points for each point to get the fitting plane.
        apply PCA to get the eigenvector with minimum value as normal of that point
        """
        _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], 10)
        k_nearest_point = points[idx, :]
        eigenvalues, eigenvectors = PCA(k_nearest_point)
        normals.append(eigenvectors[:, 2])
    # end the code

    # store the normals in the 3D point clouds
    normals = np.array(normals, dtype=np.float64)
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
