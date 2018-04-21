import numpy as np
import pandas as pd
import math

def euclud(vec1, vec2):
    return math.sqrt(sum((vec1 - vec2) ** 2))

def rand_cent(data_set, k):
    '''构建瘯质心'''
    global centroids
    n = len(data_set.columns)
    for j in range(n):
        vec = data_set[data_set.columns[j]]
        range_j = vec.max() - vec.min()
        rand_v = pd.DataFrame(vec.min() + np.random.rand(k)*range_j)
        centroids = pd.DataFrame()
        centroids = centroids.append(rand_v)
    return centroids

def k_means(data_set, k):
    m = len(data_set.columns)
    cent = rand_cent(data_set, k) #df
    cluster_as = pd.DataFrame(np.zeros((m,2)))
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = math.inf #无穷大
            min_index = -1
            for j in range(k):
                dist_ji = euclud(cent[j], data_set.loc(i))
                if dist_ji < min_dist:
                   min_dist = dist_ji; min_index = j
            if cluster_as.loc(i) != min_index:
                cluster_changed = True
                cluster_as.loc[i] = np.array([min_index, min_dist])
        print(cent)
        for c in range(k):
            '''更新质心位置'''
            if cluster_as.loc[c][0] != 0.0:
                pts_in_clust = data_set[cluster_as[0] == c]
                cent[c] = pts_in_clust.mean()