# implement Gaussian Mixture Model

import numpy as np
from numpy import *
import pylab
import random,math
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

import KMeans as km

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter


    def getLog(self, X, Pi, Mu, Var):
        N, K = len(X), len(Pi)
        p = np.zeros((N, K))

        for label in range(K):
            p[:, label] = Pi[label] * multivariate_normal.pdf(X, Mu[label], np.diag(Var[label]))

        return np.mean(np.log(p.sum(axis=1)))


    def plotClusters(self, X, Mu, Var, Mu_true=None, Var_true=None):
        colors = ['b', 'g', 'r']
        n_clusters = len(Mu)
        plt.figure(figsize=(10, 8))
        plt.axis([-10, 15, -5, 15])
        plt.scatter(X[:, 0], X[:, 1], s=5)
        ax = plt.gca()

        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls':':'}
            ellipse = Ellipse(Mu[i], 3 * Var[i][0][0], 3 * Var[i][1][1], **plot_args)
            ax.add_patch(ellipse)

        if (Mu_true is not None) and (Var_true is not None):
            for i in range(n_clusters):
                plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': '0.5'}
                ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
                ax.add_patch(ellipse)

        plt.show()


    # start the code

    """ ******************************** E step ********************************
        Update gamma (=P(z|x)): the posterior probability that a sample (data point) belongs to a cluster, NxK
        
        Given GMM parameters below, we are able to compute gamma
        
        X: input data point, NxD
        Pi: the prior probability of Gaussian distribution, Kx1
        Mu: mean of Gaussian distribution, KxD
        Var: covariance matrix of Gaussian distribution, KxDxD
    """

    def updateGamma(self, X: np.array, Mu, Var, Pi) -> np.array:
        # the number of samples / the number of features or dimensions or clusters
        N = len(X)
        K = len(Pi)
        p = np.zeros((N, K))  # pdf NxK

        for i in range(K):
            var = Var[i, :, :] # KxDxD
            p[:, i] = multivariate_normal.pdf(X, Mu[i, :], var, allow_singular=True) * Pi[i]

        # np.reshape(-1, 1) means that the row will be computed automatically, and the column is fixed
        gamma_sum = np.sum(p, axis=1).reshape(-1, 1)  # Nx1
        gamma = p / gamma_sum  # n_points*n_clusters

        return gamma

    """
    ******************************** M step ********************************
        Given GMM parameters below, we are able to compute gamma
        However, how can we estimate these parameters and gamma given data points?
        We use Maximum Likelihood to update Pi, Mu, Var. 
    """


    """
        update Pi: compute N_k and the weight of each cluster/Gaussian distribution
        gamma: posterior probability, weights through E-step
        Pi: Pi=N_k/N, the weight of each cluster
    """
    def updatePi(self, gamma):
        self.Nk_ = np.sum(gamma, axis=0) # the effective number of points assigned to cluster k, 1xK
        Pi = self.Nk_ / np.sum(gamma) # 1xK
        return Pi.reshape(-1,1) # Kx1


    """
        update Mu: the centers of k clusters
        input:
                X: data points, NxD
                gamma: posterior probability, weights through E-step, NxK
        
        return: 
                Mu: weighted average, mean of the gaussian distribution, namely, the center of the cluster
                    KxD
    """
    def update_mu(self, X, gamma):
        n_clusters = self.n_clusters
        Mu = np.zeros((n_clusters, X.shape[1]))  # KxD
        Mu = np.dot(gamma.T, X)
        Mu = Mu / self.Nk_.reshape(-1, 1)
        return Mu


    """
        update Var (covariance matrix): KxDxD
        input:
                X: data points, NxD
                Mu: weighted average, mean of the gaussian distribution, namely, the center of the cluster, KxD
                gamma: posterior probability, weights through E-step, NxK
        return: 
                Var: covariance matrix, KxDxD
    """
    def update_var(self, X, Mu, gamma):
        D = X.shape[1]
        K = self.n_clusters
        Var = np.zeros((K, D, D))  # Var: KxDxD

        for i in range(K):
            deviation = X - Mu[i, :]  # NxD
            A = np.diag(gamma[:, i])
            Var[i, :, :] = np.dot(deviation.T, np.dot(A, deviation)) / self.Nk_[i]  # var = U_T*A*U
        return Var

    # end the code

    """
        Gaussian Mixture Model Maximum Likelihood Estimation (MLE) Process:
        1. Initialize the means-Mu, covariance matrix-Var, weights-pi
        2. E-step: evaluate the posterior (gamma)
        3. M-step: estimate the parameters using MLE
        4. Evaluate the log likelihood, if converge, stop the iteration. Otherwise, repeat step 2-4.
    """
    def fit(self, data):
        # data: NxD
        # hw3
        # start the code

        """ ********************* Step 1: Initialization *********************"""
        n_clusters = self.n_clusters
        n_points = len(data)
        D = data.shape[1]

        # select k initial centers using Kmeans
        kmean = km.K_Means(n_clusters=n_clusters, max_iter=30)
        kmean.fit(data)

        # initialize GMM parameters
        Mu = kmean.cluster_centers # dimension: KxD
        Var = np.asarray([np.cov(data, rowvar=False)] * n_clusters)  # dimension: KxDxD
        pi = [1 / n_clusters] * n_clusters  # weight of each cluster： pi =[1/k, 1/k, 1/k], dimension: Kx1
        # gamma: the posterior probability that a sample (data point) belongs to a cluster
        gamma = np.ones((n_points, n_clusters)) / n_clusters   # dimension: NxK

        # iteration parameters
        log_p, old_log_p = 1, 0
        loglh = []
        time_gamma, time_pi, time_mu, time_var = 0, 0, 0, 0

        for i in range(self.max_iter):
            # self.plotClusters(X, Mu, Var)
            old_log_p = log_p

            """ ********************* Step 2: E step *********************"""
            # Update gamma: the posterior probability that a sample (data point) belongs to a cluster
            time_start = time.time()
            gamma = self.updateGamma(data, Mu, Var, pi)  #
            time_gamma += time.time() - time_start

            """ ********************* Step 3:M step *********************"""
            # update pi: the weight of each cluster/Gaussian distribution
            time_start = time.time()
            pi = self.updatePi(gamma)
            time_pi += time.time() - time_start

            # update Mu: the centers of k clusters
            time_start = time.time()
            Mu = self.update_mu(data, gamma)
            time_mu += time.time() - time_start

            # update Var: the covariance matrix of gaussian distribution
            time_start = time.time()
            Var = self.update_var(data, Mu, gamma)
            time_var += time.time() - time_start

            """ ********************* Step 4: Evaluate log likelihood *********************"""
            log_p = self.getLog(data, pi, Mu, Var)
            # loglh.append(log_p)
            # print('log-likelihood:%.3f'%loglh[-1])
            # if converged, stop the iteration
            if abs(log_p - old_log_p) < 0.001:
                break

        # update MLE parameters
        self.gamma = gamma
        self.pi = pi
        self.Mu = Mu
        self.Var = Var
        print("time:", time_gamma, time_pi, time_mu, time_var)

        # end the code


    """
        Based on the gaussian posterior (NxK), the labels with highest probability are achieved for all the data points
    """
    def predict(self, data):
        # start the code

        result = []
        gamma = self.updateGamma(data, self.Mu, self.Var, self.pi) # dimension: NxK
        label = np.argmax(gamma, axis=1)

        return label # dimension: Nx1

        # end the code


""" ************** Generate Simulated Data *************** """
def generate_X(true_Mu, true_Var):
    # First Cluster data
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)

    # Second Cluster data
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)

    # Third Cluster data
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)

    # Concatenate three clusters
    X = np.vstack((X1, X2, X3))

    # Display the data
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()

    return X

"""
    Display the classification results to check whether the data are classified correctly
    
    Input:
            label: label achieved by predict function
            X: raw data
"""
def show_cluster(label, X):
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    C1 = []
    C2 = []
    C3 = []

    for i, x in enumerate(X):
        if label[i] == 0:
            C1.append(x)
        if label[i] == 1:
            C2.append(x)
        if label[i] == 2:
            C3.append(x)

    k1 = np.array(C1)
    k2 = np.array(C2)
    k3 = np.array(C3)

    plt.scatter(k1[:, 0], k1[:, 1], s=5)
    plt.scatter(k2[:, 0], k2[:, 1], s=5)
    plt.scatter(k3[:, 0], k3[:, 1], s=5)
    plt.show()

    return X

if __name__ == '__main__':
    # Generate the data, two dimensional gaussian distribution
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    # GMM
    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    labels = gmm.predict(X)
    # print(labels)
    show_cluster(labels, X) # display predicted result

    # spectral_clustering = sc.SC(n_clusters=3, knn_k=5)
    # spectral_clustering.fit(X)
    # label = spectral_clustering.predict(X)
    # print(label)
    # show_cluster(label, X)

    

