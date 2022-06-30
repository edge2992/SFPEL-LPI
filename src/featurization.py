from sklearn.metrics import pairwise_distances
import numpy as np


def fast_LNC_calculate(X, neighbor_num):
    iteration_max = 50
    mu = 0

    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = pairwise_distances(X, metric="euclidean", n_jobs=2) + np.diag(
        e * np.inf
    )

    si = np.argsort(
        distance_matrix,
        axis=1,
    )
    nearst_neighbor_matrix = np.zeros((row_num, row_num))
    index = si[:, :neighbor_num]
    # 近傍N個をラベルづけする
    for i in range(neighbor_num):
        nearst_neighbor_matrix[i, index[i, :]] = 1

    C = nearst_neighbor_matrix
    np.random.seed(1337)
    W = np.random.rand(row_num, row_num).T  # matlabと乱数を合わせる
    W = C * W
    _lambda = 8 * e
    P = X @ X.T + _lambda @ e.T
    for i in range(iteration_max):
        Q = (C * W) @ P + mu * (C * W)
        W = (C * W) * P / Q
        W[np.isnan(W)] = 0
    return W
