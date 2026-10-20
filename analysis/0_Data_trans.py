# -*- coding: utf-8 -*-
"""
SemST 完整数据流: 打印每个模块的输入 / 输出维度
=====================================================================
数据流总览:
  原始数据 → 基因筛选&归一化 → 特征图 fadj / 空间图 sadj
       │
       └→ LLM 语义向量 (N × 2560)
  ──进入模型──
  SGCN / FGCN → 拼接 → LLM Head → 调制 MLP → FSM 特征融合
  → 融合 MLP → latent_emb → ZINB 解码器 → KMeans 聚类
=====================================================================
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scanpy as sc
from sklearn.cluster import KMeans

from models import SemST
from preprocess.config import Config
from utils import normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor

DATASET = 'Mouse_Brain_Anterior'


#   DLPFC     -> 文件夹, 10x Visium,          读 read_visium, 标签 layer_guess_reordered
#   Embryo    -> 单文件 E9.5_{id}.MOSTA.h5ad,  读 read_h5ad,   标签 annotation
#   HBC / MBA -> 文件夹, 10x Visium,          读 read_visium, 标签 ground_truth
#   MVC       -> 单文件 adata.h5ad,            读 read_h5ad,   标签 cluster
DLPFC_SECTIONS = ['151507', '151508', '151509', '151510',
                  '151669', '151670', '151671', '151672',
                  '151673', '151674', '151675', '151676'] # 当前只有151508
EMBRYO_SECTIONS = ['E1S1']


def resolve_dataset(dataset_id):
    """根据数据集名解析出 配置名 / 预处理后的 h5ad 路径 / 原始数据路径 / 读取方式 / 标签列。

    返回 dict, 字段: config, h5ad, raw, reader, label_col, raw_format
    """
    if dataset_id in DLPFC_SECTIONS:
        return dict(config='DLPFC',
                    h5ad=f'generate_data/DLPFC/{dataset_id}/data.h5ad',
                    raw=f'data/raw/DLPFC/{dataset_id}',            # 文件夹
                    reader='read_visium',
                    label_col='layer_guess_reordered',
                    raw_format='文件夹 (10x Visium, read_visium)')

    if dataset_id in EMBRYO_SECTIONS:
        return dict(config='Embryo',
                    h5ad=f'generate_data/Embryo/{dataset_id}/data.h5ad',
                    raw=f'data/raw/Embryo/E9.5_{dataset_id}.MOSTA.h5ad',  # 带 E9.5_ 前缀 / .MOSTA 后缀
                    reader='read_h5ad',
                    label_col='annotation',
                    raw_format='单文件 E9.5_{id}.MOSTA.h5ad (read_h5ad)')

    if dataset_id == 'MVC':
        return dict(config='MVC',
                    h5ad='generate_data/MVC/data.h5ad',
                    raw='data/raw/MVC/adata.h5ad',                  # 固定文件名 adata.h5ad
                    reader='read_h5ad',
                    label_col='cluster',
                    raw_format='单文件 adata.h5ad (read_h5ad)')

    # Human_Breast_Cancer / Mouse_Brain_Anterior 等: 文件夹(Visium)
    return dict(config=dataset_id,
                h5ad=f'generate_data/{dataset_id}/data.h5ad',
                raw=f'data/raw/{dataset_id}',                        # 文件夹
                reader='read_visium',
                label_col='ground_truth',
                raw_format='文件夹 (10x Visium, read_visium)')


def show_layer(name, layer, extra=''):
    """打印 线性层 / 图卷积层 的 输入→输出 维度"""
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        print(f'    {name}: 输入 {layer.in_features} 维 → 输出 {layer.out_features} 维 {extra}')


def shape(t):
    """把张量/矩阵的 shape 变成统一的可打印字符串"""
    return str(tuple(t.shape))


ds_info = resolve_dataset(DATASET)
config_name = ds_info['config']
h5ad_path = ds_info['h5ad']

print('=' * 78)
print(f'                SemST 完整数据流 · 各模块输入输出维度')
print(f'                当前数据集: {DATASET}  (配置文件: {config_name}.ini)')
print('=' * 78)

# =====================================================================
# 【模块 1】LLM 语义向量生成  (llm_generate_embedding.py)
#   每个 spot 取表达量最高的 10 个基因名 → 拼成一句话 → 喂给 LLM
#   → 对 token 隐状态做掩码平均池化 → 得到每个 spot 的语义向量
# =====================================================================
print('\n【模块 1】LLM 语义向量 (llm_generate_embedding.py)')
print('  流程: 基因名文本 → tokenizer → LLM(Qwen3-4B, 隐藏维 2560) → 掩码平均池化')
print('  每个 spot 的输出维度 = LLM 隐藏层维度 (理论值 2560), 实际如下:')
for name in ['Mouse_Brain_Anterior', 'MVC', 'Human_Breast_Cancer', 'E1S1',
             '151507', '151508']:
    p = os.path.join(ROOT, f'data/npys/embeddings_{name}.npy')
    if os.path.exists(p):
        a = np.load(p)
        print(f'    {name:<22} 形状 {a.shape}  (spot 数 × LLM 维度 {a.shape[-1]})  dtype={a.dtype}')

# =====================================================================
# 【模块 2】数据读取与预处理  (preprocess/*_generate_data.py 的 normalize)
#   原始计数矩阵 → 基因过滤(min_cells=100) → 选 HVG(fdim 个) → CPM 归一化 → scale
# =====================================================================
print('\n【模块 2】数据读取与预处理 (normalize)')
print(f'  原始数据(格式因数据集而异, 见 preprocess/*_generate_data.py):')
print(f'    路径:     {ds_info["raw"]}')
print(f'    读取方式: {ds_info["raw_format"]}')
print(f'    标签列:   {ds_info["label_col"]}')
print(f'  ↓ 预处理后统一写入: {ds_info["h5ad"]}')
print(f'  输入: 原始基因计数矩阵 (N_spot × N_全基因)')
adata = sc.read_h5ad(os.path.join(ROOT, h5ad_path))
features = torch.FloatTensor(adata.X)          # 模型真正吃的表达矩阵
print(f'  输出: adata.X (表达矩阵)   {shape(adata.X)}   (N_spot × fdim 高变基因)')
print(f'        n_obs(spot 数)={adata.n_obs}, n_vars(基因数)={adata.n_vars}')
print(f'        spatial(空间坐标)    {shape(adata.obsm["spatial"])}   (N_spot × 2)')
print(f'        ground(真实标签类别数) = {len(np.unique(adata.obs["ground"]))}')

# =====================================================================
# 【模块 3】图构建  (preprocess/construction.py)
#   特征图 fadj: 用基因表达找相似邻居 (knn, 余弦距离)
#   空间图 sadj: 用空间坐标找邻居 (k近邻 / 半径 / 自适应)
# =====================================================================
print('\n【模块 3】图构建 (construction.py)')
fadj = adata.obsm['fadj']
sadj = adata.obsm['sadj']
graph_nei = adata.obsm['graph_nei']
graph_neg = adata.obsm['graph_neg']
print(f'  输入: 特征矩阵 (N × fdim)          -> 输出 fadj       {shape(fadj)}   (N × N 稀疏)')
print(f'  输入: 空间坐标 (N × 2)             -> 输出 sadj       {shape(sadj)}   (N × N 稀疏)')
print(f'                                       输出 graph_nei  {shape(graph_nei)} (正邻居)')
print(f'                                       输出 graph_neg  {shape(graph_neg)} (负邻居)')

# =====================================================================
# 【模块 4】邻接矩阵归一化  (utils.py)
#   +单位阵(自连接) → 行归一化 → 转 PyTorch 稀疏张量
# =====================================================================
print('\n【模块 4】邻接矩阵归一化 (normalize_sparse_matrix)')
N = adata.n_obs
nfadj = normalize_sparse_matrix(fadj + sp.eye(N))
nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
nsadj = normalize_sparse_matrix(sadj + sp.eye(N))
nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
print(f'  输入: fadj/sadj (N × N) + 单位阵  -> 输出 nfadj/nsadj  {shape(nfadj)}  (PyTorch 稀疏张量)')

# =====================================================================
# 【模块 5】SemST 模型各模块  (models.py)
#   逐个打印每个子模块的 输入→输出 维度, 并做一次真实前向验证
# =====================================================================
print('\n【模块 5】SemST 模型各子模块 (models.py)')

config = Config(os.path.join(ROOT, f'config/{config_name}.ini'))
fdim = config.fdim     # 输入基因数
nhid1 = config.nhid1   # GCN 第一层隐藏维
nhid2 = config.nhid2   # GCN 第二层输出维 / 最终 embedding 维
ratio = config.llm_modulation_ratio

# 加载该数据集的 LLM 语义向量, 得到实际维度
llm_emb = torch.tensor(
    np.load(os.path.join(ROOT, f'data/npys/embeddings_{DATASET}.npy')),
    dtype=torch.float32)
llm_dim = llm_emb.shape[-1]

print(f'  超参数: fdim={fdim}, nhid1={nhid1}, nhid2={nhid2}, '
      f'llm_dim={llm_dim}, llm_modulation_ratio={ratio}')
print(f'  模型输入张量: x {shape(features)} | sadj/fadj {shape(nsadj)} | llm_emb {shape(llm_emb)}')

model = SemST(nfeat=fdim, nhid1=nhid1, nhid2=nhid2, dropout=config.dropout,
              llm_dim=llm_dim, llm_modulation_ratio=ratio, use_llm=True)
model.eval()

print('  ── 各层参数量对应的 输入→输出 维度 ──')
print('  [SGCN 空间编码器]  (吃 sadj 空间邻接图)')
show_layer('gc1  GraphConvolution', model.SGCN.gc1)
show_layer('gc2  GraphConvolution', model.SGCN.gc2)
print('  [FGCN 特征编码器]  (吃 fadj 特征邻接图)')
show_layer('gc1  GraphConvolution', model.FGCN.gc1)
show_layer('gc2  GraphConvolution', model.FGCN.gc2)
print('  [LLM Head]  把 LLM 向量映射到与 GCN 输出相同维度')
show_layer('mlp[0] Linear', model.llm_head.mlp[0], '(ReLU 后接 mlp[2])')
show_layer('mlp[2] Linear', model.llm_head.mlp[2])
print('  [调制 MLP]  由语义向量生成 FSM 的缩放/偏移参数')
show_layer('modulation_mlp[0] Linear', model.modulation_mlp[0], '(ReLU+Dropout 后接 [3])')
show_layer('modulation_mlp[3] Linear', model.modulation_mlp[3])
print('  [融合 MLP]  融合后的嵌入压缩回最终 latent 维度')
show_layer('MLP[0] Linear', model.MLP[0], '(ReLU+Dropout 后接 [3])')
show_layer('MLP[3] Linear', model.MLP[3])
print('  [ZINB 解码器]  把 latent 映射回原始基因表达')
show_layer('decoder[0] Linear', model.ZINB.decoder[0], '(BatchNorm+ReLU 后接 pi/disp/mean)')
show_layer('pi    Linear', model.ZINB.pi, '(→ sigmoid)')
show_layer('disp  Linear', model.ZINB.disp, '(→ softplus)')
show_layer('mean  Linear', model.ZINB.mean, '(→ exp)')

print('\n  ── 真实前向传播, 逐模块打印张量维度 ──')
with torch.no_grad():
    emb1 = model.SGCN(features, nsadj)
    print(f'  SGCN: x {shape(features)} -> emb1 {shape(emb1)}')
    emb2 = model.FGCN(features, nfadj)
    print(f'  FGCN: x {shape(features)} -> emb2 {shape(emb2)}')

    emb_combined = torch.cat((emb1, emb2), dim=1)
    print(f'  拼接: cat(emb1, emb2) -> emb_combined {shape(emb_combined)}  (2 × nhid2)')

    processed_llm = model.llm_head(llm_emb)
    print(f'  LLM Head: llm_emb {shape(llm_emb)} -> processed_llm {shape(processed_llm)}')

    modulation = model.modulation_mlp(processed_llm)
    print(f'  调制 MLP: {shape(processed_llm)} -> modulation_params {shape(modulation)}  (4 × nhid2)')

    alpha, beta = torch.chunk(modulation, 2, dim=1)
    print(f'  切分: chunk(modulation, 2) -> alpha {shape(alpha)}, beta {shape(beta)}  (各 2 × nhid2)')

    emb = (1 + alpha) * emb_combined + beta
    print(f'  FSM 融合: (1+alpha)*emb_combined + beta -> emb {shape(emb)}  (2 × nhid2)')

    latent_emb = model.MLP(emb)
    print(f'  融合 MLP: {shape(emb)} -> latent_emb {shape(latent_emb)}  (最终用于聚类的嵌入)')

    pi, disp, mean = model.ZINB(latent_emb)
    print(f'  ZINB 解码: latent_emb {shape(latent_emb)} ->')
    print(f'      pi   {shape(pi)}  (dropout 概率, sigmoid)')
    print(f'      disp {shape(disp)} (离散度, softplus)')
    print(f'      mean {shape(mean)} (均值, exp)')

# =====================================================================
# 【模块 6】KMeans 聚类  (train.py / load_model_clustering.py)
# =====================================================================
print('\n【模块 6】KMeans 聚类')
latent_np = latent_emb.numpy()
kmeans = KMeans(n_clusters=config.class_num if hasattr(config, 'class_num') else len(np.unique(adata.obs['ground'])),
                n_init='auto', random_state=config.seed).fit(latent_np)
print(f'  输入: latent_emb {shape(latent_np)}  -> 输出: 聚类标签 {shape(kmeans.labels_)}  (每个 spot 一个类)')

# =====================================================================
# 数据流维度总览
# =====================================================================
print('\n' + '=' * 78)
print('                        数据流维度总览 (逐层)')
print('=' * 78)
print(f'  原始基因计数矩阵          (N × N_全基因)')
print(f'  └─ 筛选 HVG + 归一化      → 表达矩阵 x        {shape(adata.X)}')
print(f'  └─ 建图                    → fadj / sadj       {shape(fadj)}')
print(f'  └─ LLM 语义向量            → llm_emb           {shape(llm_emb)}')
print(f'  SGCN / FGCN 编码            → emb1 / emb2       {shape(emb1)}')
print(f'  拼接                        → emb_combined     {shape(emb_combined)}')
print(f'  LLM Head                    → processed_llm    {shape(processed_llm)}')
print(f'  调制 MLP                    → modulation       {shape(modulation)}')
print(f'  FSM 特征融合                → emb              {shape(emb)}')
print(f'  融合 MLP                    → latent_emb       {shape(latent_emb)}   ← 聚类用')
print(f'  ZINB 解码                   → pi/disp/mean     {shape(pi)}')
print(f'  KMeans                      → 聚类标签         {shape(kmeans.labels_)}')
print('=' * 78)
