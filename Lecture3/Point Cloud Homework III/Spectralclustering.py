import numpy as np
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt


def distance(p1, p2):
    """ Get the distance between two points """
    dist = np.sqrt(np.power(p1 - p2, 2).sum())
    return dist

def get_dist_matrix(data):
    """
        Get the distance matrix
        input: raw data
        return: distance matrix
    """
    n = len(data)  # dimension: NxD
    # initialize distance matrix, dimension: NxN
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i][j] = dist_matrix[j][i] = distance(data[i], data[j])
    return dist_matrix

class SC(object):

    def __init__(self, n_clusters, knn_k):
        self.n_clusters  = n_clusters
        self.knn_k = knn_k

    """ 
        Get adjacent matrix
        input:
                data: raw data
                k: the number of cluster
        return:
                adjacent matrix
    """
    def getW(self, data, k):
        n = len(data)
        dist_matrix = get_dist_matrix(data)

        W = np.zeros((n, n))
        for idx, dist in enumerate(dist_matrix):
            # sort each row and get index list
            # smaller distance means two points are closer
            idx_array = np.argsort(dist)
            # set the element in each row to 1
            # except for the diagonal elements
            W[idx][idx_array[1 : k + 1]] = 1
        W_T = np.transpose(W)
        return (W + W_T) / 2

    """
        Get degree matrix
        input:
              W: adjacent matrix
        return:
              degree matrix
    """
    def getD(self, W):
        D = np.diag(sum(W))
        return D

    """
        Get unnormalized Laplace matrix
        input:
              W: adjacent matrix
              D: degree matrix
        return:
              Laplace matrix
    """
    def getL(self, D,W):
        return D-W

    """
           Get eigen matrix of Laplace matrix
           input:
                 L: Laplace matrix
                 k: the number of clusters
           return:
                 eigen matrix
       """
    def getEigen(self, L, cluster_num):
        eig_vec, eig_val, _ = np.linalg.svd(L)
        # get the first k smallest eigenvectors
        idx = np.argsort(eig_val)[0 : cluster_num]
        return eig_vec[:, idx]


    def fit(self, data):
        k = self.knn_k
        cluster_num = self.n_clusters
        data = np.array(data)
        W = self.getW(data, k)
        D = self.getD(W)
        L = self.getL(D, W)
        eig_vec = self.getEigen(L, cluster_num)
        self.eigvec = eig_vec


    def predict(self, data):
        clf = KMeans(n_clusters=self.n_clusters)
        s = clf.fit(self.eigvec)  # clusters
        labels = s.labels_
        return  labels


if __name__ == '__main__':
    cluster_num = 3
    knn_k = 5
    data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    data = data[0:-1]  # last column is the label
    spectral_clustering = SC(n_clusters= 3, knn_k = 5)
    spectral_clustering.fit(data)
    label = spectral_clustering.predict(data)

