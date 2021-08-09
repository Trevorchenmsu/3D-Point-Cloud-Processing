# implementation of K-means algorithm

import numpy as np
import random

# get the minimum distance between a point and several centroids
def get_closest_distance(point, centroids):
    min_dist = np.inf

    for i, centroid in enumerate(centroids):
        dist = np.sum((np.array(point) - np.array(centroid)) ** 2) # standard deviation
        min_dist = min(dist, min_dist)
    return min_dist

# Roulette Wheel Selection: initialize the centers of k clusters
def get_initial_cluster_centers(data: np.array, k: int) -> list:
    cluster_centers = []
    data = list(data)
    cluster_center = random.choice(data)
    cluster_centers.append(cluster_center)
    distances = [0 for _ in range(len(data))]

    for _ in range(1, k):
        total = 0.0

        for i, point in enumerate(data):
            distances[i] = get_closest_distance(point, cluster_centers) # the distance from a closest cluster center
            total += distances[i]

        total *= random.random()

        for i, dist in enumerate(distances): # select a next cluster center
            total -= dist
            if total > 0: continue
            cluster_centers.append(data[i])
            break

    return cluster_centers

class K_Means(object):
    # k: the number of clusters; tolerance: tolerance to the central point; max_iter: maximum iterations
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.cluster_centers = []

    """
        Based on the dataset, output the K cluster centers
        input: 
              data: raw data, N * D (N: the number of samples, D: the dimension of data point)
    """
    def fit(self, data):
        # hw1
        # start the code

        # **************** step 1: initialize cluster centers  ****************
        if data is None:
            return False

        k = self.k_
        N, D = data.shape

        data_centers = np.zeros((k, D))
        new_data_centers = np.zeros((k, D))

        # randomly select k cluster centers
        # seed_idx = random.sample(list(range(N)), k)
        # data_centers = data[seed_idx, :]

        # an optimized method to select k cluster center
        data_centers = get_initial_cluster_centers(data, k)

        print('----------------Initial K Cluster Centers ---------------')
        print(data_centers)

        # **************** step 2: EM STEP  ****************
        iteration = 0
        tolerance = np.inf
        loss = 1000

        while tolerance > self.tolerance_ and iteration < self.max_iter_:
            # -------------- Expectation Step (E Step) ---------------
            label_idx = np.zeros((N, 1), dtype=int)

            # find the closest cluster center for each point. Brute force method
            # The process can be improved by applying kdtree or octree search
            for i in range(N):
                min_dist = np.inf
                label = 0 # label for the cluster
                for j in range(k):
                    dist = np.sum((data[i] - data_centers[j]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        label = j

                # this indicates that point_i belongs to j(label) cluster
                label_idx[i, :] = label

            # divide data into different clusters
            new_data = np.hstack((label_idx, data))

            # -------------- Maximization Step (M Step) ---------------
            # compute the mean of the current cluster as the new cluster center
            for cluster_label in range(k):
                count_points_cluster = 0
                for i in range(N):
                    if new_data[i, 0] == cluster_label:
                        new_data_centers[cluster_label, :] += new_data[i, 1 : k + 1] # K * D
                        count_points_cluster += 1
                new_data_centers[cluster_label, :] = new_data_centers[cluster_label, :] \
                                                     / count_points_cluster

            # **************** step 3: State Update  ****************
            loss_old = loss
            loss = np.sum(np.linalg.norm(new_data_centers - data_centers, axis = 1))

            data_centers = new_data_centers.copy()
            tolerance = abs(loss - loss_old)
            iteration += 1

        self.cluster_centers = data_centers.copy()

        # end the code

    """
        Classification: cluster the dataset to k clusters
        input: 
              p_data: input data, N x D, D should be same as the number of classes 
    """

    def predict(self, p_data):
        result = []
        # hw2
        # start the code

        if p_data is None:
            return False

        N, D = p_data.shape
        k = self.k_

        data_centers = self.cluster_centers.copy()

        for i in range(N): # find a closest center for each point
            min_dist = np.inf
            label = 0
            for j in range(k):
                # compute the distance between ith point and jth cluster center
                dist = np.linalg.norm(p_data[i, :] - data_centers[j, :])
                if dist < min_dist:
                    min_dist = dist
                    label = j
            result.append(label)

        # end the code
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
    










