import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMHead(nn.Module):
    """把LLM向量映射到与GCN输出相同的维度
    llm_dim -> nhid2
    """
    def __init__(self, input_dim, output_dim, hidden_dim_ratio=0.5):
        super().__init__()

        hidden_dim = max(1, int(input_dim * hidden_dim_ratio))
        # 两层MLP
        # 先 Linear(2560→1280)，再 ReLU 激活，再 Linear(1280→64)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    output = adj @ x @ W, 加上邻居点信息
    相邻的点因得到邻居的特征变得相似，使得同一类的点在特征空间中聚集在一起，便于后续分类器的分类
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # nn.Parameter:可训练参数。训练时会用梯度下降自动更新。
        # 要学习的权重
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # 要学习的偏置
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) # 开方分之一，这样初始值不会太大，训练稳定
        self.weight.data.uniform_(-stdv, stdv) # 权重随机设成之间的小数
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight) # 先做普通线性变换，只用自己的信息
        output = torch.spmm(adj, support) 
        # 再做邻接矩阵的稀疏矩阵乘法（只存1）
        # 这里adj以及进行了行归一化，变成了邻居的加权平均
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """两层GCN编码器，二阶邻居，看到更大范围空间的上下文"""
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # 第一层，同时加非线性
        x = F.dropout(x, self.dropout, training=self.training) # 随机丢一些神经元防过拟合
        x = self.gc2(x, adj) # 第二层，输出最终的embedding
        return x


class ZINBDecoder(torch.nn.Module):
    """
    ZINB解码器，输出pi, disp, mean
    和编码器任务相反，将64维嵌入映射回原始基因表达
    """
    def __init__(self, nfeat, nhid1, nhid2):
        """
        pi: dropout概率，表示基因表达为0的概率
        disp: 离散参数，表示基因表达的离散程度
        mean: 均值参数，如果检测到了，平均表达量是多少
        """
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb) # (N,64)->(N,128)->(N,128)
        pi = torch.sigmoid(self.pi(x)) # (N,3000)，压缩到(0,1)
        disp = self.DispAct(self.disp(x)) # (DispAct)softplus: log(1+exp(x))，保证disp>0
        mean = self.MeanAct(self.mean(x)) # (MeanAct)exp:保证mean>0
        return [pi, disp, mean] 
        # 交给utils.py里的ZINBLoss计算损失, 和x算重建损失


class SemST(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, llm_dim, llm_modulation_ratio, use_llm=True):
        super(SemST, self).__init__()

        # 吃sadj（空间邻接图：空间上相邻的点才连边）
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        # 吃fadj（特征邻接图：基因表达相似的点才连边）
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = ZINBDecoder(nfeat, nhid1, nhid2)
        self.dropout = dropout

        self.use_llm = use_llm
        if use_llm:
            self.llm_head = LLMHead(llm_dim, nhid2)

            mod_mlp_input_dim = nhid2
            # 4倍
            mod_mlp_output_dim = 4 * nhid2
            mod_mlp_hidden_dim = max(1, int(mod_mlp_input_dim * llm_modulation_ratio))
            self.modulation_mlp = nn.Sequential(
                nn.Linear(mod_mlp_input_dim, mod_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mod_mlp_hidden_dim, mod_mlp_output_dim)
            )

        # 融合层：64维嵌入映射回128维，再映射回64维
        self.MLP = nn.Sequential(
            nn.Linear(nhid2 * 2, nhid1), # 128 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid1, nhid2) # 128 -> 64
        )

    def forward(self, x, sadj, fadj, llm_emb):
        emb1 = self.SGCN(x, sadj)
        emb2 = self.FGCN(x, fadj)

        emb_combined = torch.cat((emb1, emb2), dim=1) # 拼接成 (N, 128)

        if self.use_llm:
            processed_llm_emb = self.llm_head(llm_emb)
            modulation_params = self.modulation_mlp(processed_llm_emb)

            alpha, beta = torch.chunk(modulation_params, 2, dim=1)
            emb = (1 + alpha) * emb_combined + beta # FSM: 融合GCN嵌入和LLM嵌入
            # alpha: 控制GCN嵌入的缩放
            # beta: 控制GCN嵌入的偏移
        else:
            emb = emb_combined

        latent_emb = self.MLP(emb) # 用来聚类的最终嵌入，(N, 64)
        [pi, disp, mean] = self.ZINB(latent_emb)

        return latent_emb, pi, disp, mean, emb1, emb2
        # emb1: SGCN嵌入，emb2: FGCN嵌入，给dicr_loss去做去相关(得到互补而非重复的信息)
