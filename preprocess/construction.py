import numpy as np
import ot
import pandas as pd
import scipy.sparse as sp # 稀疏邻接矩阵
import sklearn
import torch
from sklearn.metrics.pairwise import euclidean_distances # 欧氏距离
from sklearn.neighbors import kneighbors_graph # k近邻图

"""
    - 前面三个使用空间位置找邻居
    - 最后一个使用基因表达找邻居

    Input:
    - positions:空间坐标, N×2
    - adata:AnnData对象, 存储空间转录组数据
    - features:特征矩阵
"""

def spatial_construct_graph2(position, k=3):  # 3
    # calculate distance matrix
    """限制个数：每个细胞跟离他最近的k=3个细胞建立边，构建空间图"""
    distance_matrix = ot.dist(position, position, metric='euclidean') # 计算欧氏距离矩阵
    n_spot = distance_matrix.shape[0]
    # print("n_spot:", n_spot)

    # adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot]) # 记录哪些点之间存在连接
    for i in range(n_spot):
        vec = distance_matrix[i, :] # 第i个细胞到其他所有细胞的距离
        distance = vec.argsort() # 按从小到大排列，返回下标
        for t in range(1, k + 1):
            y = distance[t]
            interaction[i, y] = 1

    # 目前还是有向图
    adj = interaction
    adj = adj + adj.T # 加上转置，自己和别人的加起来
    adj = np.where(adj > 1, 1, adj) # 超过1，就统一变成1，保证是无向图

    graph_nei = torch.from_numpy(adj) # NumPy邻接矩阵转换为PyTorch张量
    graph_neg = torch.ones(position.shape[0], position.shape[0]) - graph_nei

    # 将 adj 转换为 SciPy 的 COO 格式稀疏矩阵
    # 只存非零的坐标和元素
    sadj = sp.coo_matrix(adj, dtype=np.float32) 
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg
    # sadj: 稀疏邻接矩阵
    # graph_nei: 正图
    # graph_neg: 负图


def spatial_construct_graph1(adata, radius=150):
    """不限制个数：半径150以内都算邻居"""
    coor = pd.DataFrame(adata.obsm['spatial'])
    # print("coor.shape:", coor.shape)
    
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A = np.zeros((coor.shape[0], coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor) # 创建最近邻搜索器
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True) 
    # 每个邻居indices下标, indices[i].shape[0]:当前spot的邻居数量

    for it in range(indices.shape[0]):
        # 利用 NumPy 的高级索引，批量写入邻接矩阵。
        # 把当前 spot 与所有半径内邻居之间的边标记为 1
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg  # , nsadj


def spatial_construct_graph(positions, k=15):
    """自动：每个spot至少连接2个点，至少有一个spot连接超过15个点"""
    print("start spatial construct graph")
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100): # 起始100，终止1000（不包含），步长100
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10): # 步长10
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5): # 步长5
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    # ！存在风险：if找不到符合条件的阈值，则仍是最初欧氏距离矩阵，当作邻接矩阵使用
    row, col = np.diag_indices_from(A)
    A[row, col] = 0 # 对角线清零
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/edge.csv', index, delimiter=',')

    graph_nei = torch.from_numpy(A)
    # print(type(graph_nei),graph_nei)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg  # , nsadj


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    """按基因相似度(特征向量)构建图"""
    # connectivity：输出邻接关系，而不是距离值 1/0
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1) # -1：自动推断
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    # 使用余弦相似度，找k个，k+1包含自己
    # kneighbors_graph() 根据输入特征，为每个样本寻找最近的邻居，并生成稀疏邻接矩阵
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray() # SciPy稀疏矩阵转换为普通NumPy数组
    row, col = np.diag_indices_from(A)
    A[row, col] = 0 # 去掉自己
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    # 取里面较大的那个，一方认就算认，保证是无向图
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj  # , nfadj