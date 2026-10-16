import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

"""
    SemST的工具箱:
    1. 构造loss
    2. 处理图数据
    3. 降维
    4. 处理数值异常
    5. 处理评价指标
"""

EPS = 1e-15

def regularization_loss(emb, graph_nei, graph_neg):
    """
    计算空间约束loss:
        - 计算正负邻居的相似度损失
        - 该损失函数的目标是最大化空间邻居的相似度，
          同时最小化负邻居的相似度，从而提高嵌入表示的质量。

    Args:   
        - emb: 模型学到的 spot embedding
        - graph_nei: 空间邻接矩阵,表示相邻
        - graph_neg: 负邻接矩阵
    Returns:
        - pair_loss: 空间约束 loss
    """
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    # 对于空间邻居，希望相似度趋于1, 对于负邻居，希望相似度趋于0
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()

    # 得到空间约束 loss，是邻居但损失大 -> 被惩罚
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss


def cosine_similarity(emb):
    """
    计算嵌入向量之间的余弦相似度矩阵
    Args:
        - emb: 嵌入向量矩阵
    Returns: 
        - 余弦相似度矩阵
    """
    mat = torch.matmul(emb, emb.T) # 计算点积
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1)) # 长度
    mat = torch.div(mat, torch.matmul(norm, norm.T)) # 归一化，得到余弦相似度
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat)) # 对角线，自己与自己相似度1，去除掉
    return mat


def off_diagonal(x):
    """
    提取矩阵非对角元素，用于相关性损失
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    # 巧妙方法：先 flatten()，再[:-1]去掉最后一个元素，变成(n*n-1)个元素
    # 然后 view(n-1, n+1) 变成(n-1)*(n+1)的矩阵，最后[:, 1:]去掉第一列，剩下(n-1)*n个元素，即为非对角的


def cross_correlation(Z_v1, Z_v2):
    """
    计算交叉视图的相关矩阵 S
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    S[i][j]: "视图1的第 i 维"和"视图2的第 j 维"之间的相关性
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):
    """
    L_cr: 相关降低损失L
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def dicr_loss(com1, com2):
    """
    DICR损失
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    # 传的是转置，计算特征之间的互相关（FCR）
    S_F_ae = cross_correlation(com1.t(), com2.t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)

    # loss of DICR
    loss_dicr = L_F_ae

    return loss_dicr


def dopca(data, dim=50):
    """PCA 降维, 减少计算量"""
    return PCA(n_components=dim).fit_transform(data)


def _nan2zero(x):
    # _nan2zero：NaN → 0，用在相似度计算里（零向量相除会产生 nan，就当成"没相似度"）
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    # _nan2inf：NaN → inf，用在 loss 计算里（零向量相除会产生 nan，就当成"损失很大"，告诉模型错误）
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 将 SciPy 稀疏矩阵转换为 COO 格式
    # COO格式：记录非零元素的行索引、列索引和对应的值
    # 将 COO 格式的稀疏矩阵转换为 PyTorch 的稀疏张量 (COO+shape)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_sparse_matrix(mx):
    """
    Row-normalize sparse matrix
    进行行归一化，使得每个节点的邻居特征对其影响一致，避免度数较高的节点对模型产生过大影响
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class NB(object):
    """
    基因转录组里的基因表达可能含大量0,不适合MSE
    使用负二项分布
    """
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    """零膨胀负二项分布
    在NB基础上，零值来源于两种情况：1.真实的零值，2.由于技术原因导致的零值（dropout）    
    """
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi # 概率
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


def run_precomputation_llm_base(npy_path=None):
    """加载预计算LLM嵌入向量"""
    final_base_embeddings = np.load(npy_path)
    final_base_embeddings = torch.tensor(final_base_embeddings)
    llm_hidden_dim = final_base_embeddings.shape[-1]
    return final_base_embeddings, llm_hidden_dim


def hungarian_match(y_true, y_pred):
    """聚类是无监督的，使用匈牙利算法对 y_pred 的标签进行最佳匹配，用于后续计算ACC/F1"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_matrix[y_pred[i], y_true[i]] += 1 
        # 预测pred[i]的标签和真实y_true[i]的标签对应的计数加1

    # linear_sum_assignment：求总代价最小的配对方案
    # 我们前面得到的是总收益最大（数字最大），需进行翻转
    # 每行选一个，每列选一个，要求最小
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}

    y_pred_aligned = np.array([mapping[y] for y in y_pred])
    return y_pred_aligned
