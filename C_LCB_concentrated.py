import numpy as np

from sklearn.neighbors import NearestNeighbors


def evaluate_concentration_multi_dimension2(data, k_neighbors):  # 计算点与最邻近点的总距离之和的平均值
    """
    :param data: ndarray, 探索点のlist
    :param k_neighbors: k近傍法のパラメータ
    :return: 探索点の集中度
    """

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')

    nbrs.fit(data)

    distances, _ = nbrs.kneighbors(data)

    local_ave_distances = np.mean(distances, axis=1)

    ave_distances = np.mean(local_ave_distances)

    return ave_distances




