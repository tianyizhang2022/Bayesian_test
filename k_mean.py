import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering(data, max_cluster_radius, num_points_to_generate_per_cluster):
    """

    :param data: (ndarray) 探索点のlist
    :param max_cluster_radius: (float)cluster最大半径
    :param num_points_to_generate_per_cluster: (int) 局所近似関数誤差を計算するため生成するサンプル点数
    :return: new_clustered_points:集中空間のサンプル点list
             new_discrete_points:離散空間のサンプル点list

    """
    kmeans = KMeans(n_clusters=2)  # 设置较小的点簇数量

    # 根据最大半径进行自适应的簇数量判断
    for num_clusters in range(2, len(data)):  # 从较小的点簇数量开始尝试
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)

        # 获取每个点到簇中心的距离
        distances = kmeans.transform(data)
        min_distances_to_centers = np.min(distances, axis=1)

        if np.max(min_distances_to_centers) <= max_cluster_radius:
            break  # 如果所有点都满足最大半径限制，则退出循环

    assigned_clusters = kmeans.predict(data)
    clustered_points = []

    for i in range(num_clusters):
        cluster = data[assigned_clusters == i]
        cluster_center = kmeans.cluster_centers_[i]

        # 随机生成与簇大小相同数量的点
        if len(cluster) > 0:
            random_points = np.random.rand(num_points_to_generate_per_cluster, cluster.shape[1]) * (np.max(cluster, axis=0) - np.min(cluster, axis=0)) + np.min(cluster, axis=0)
            clustered_points.append(random_points)

    # 处理未分类到任何簇的点，将它们作为离散点
    unclassified_points = data[assigned_clusters == -1]

    # 生成额外的离散点
    random_discrete_points = np.random.rand(num_points_to_generate_per_cluster, data.shape[1]) * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)

    # 返回新生成的簇内点集合和离散点集合
    new_clustered_points = np.concatenate([cluster for cluster in clustered_points], axis=0)
    new_discrete_points = np.concatenate([unclassified_points, random_discrete_points], axis=0)

    return new_clustered_points, new_discrete_points

