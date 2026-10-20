"""
SemST 误差分析公共模块

对任意一个数据集：
    1. 加载预处理好的 data.h5ad + 训练好的权重 + 预计算的 LLM 嵌入
    2. 前向得到聚类嵌入, KMeans + 匈牙利对齐
    3. 计算 ARI / NMI / AMI / ACC / F1
    4. 逐类准确率
    5. 误差归因分析(类别规模、空间邻接混淆、稀有亚型被吞并)

被 run_*.py 调用, 每个数据集一个 runner 脚本。
"""
from __future__ import division, print_function

import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
from sklearn import metrics
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# 保证能 import 项目根目录下的 models / utils / preprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SemST                                        # noqa: E402
from preprocess.config import Config                           # noqa: E402
from utils import (                                             # noqa: E402
    normalize_sparse_matrix,
    sparse_mx_to_torch_sparse_tensor,
    run_precomputation_llm_base,
    hungarian_match,
)

# DLPFC 有多个切片, 这里用其中的一个(151508), 其余切片结构相同
DLPFC_IDS = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672']


def resolve_config(dataset_id):
    """由数据集 id 反查 config 名(与 train.py 一致)"""
    if dataset_id in DLPFC_IDS:
        return 'DLPFC'
    if dataset_id == 'E1S1':
        return 'Embryo'
    return dataset_id


def load_data(dataset_id, config_name):
    """读取预处理数据, 返回归一化后的图与标签"""
    prefix = config_name + "/" if config_name in ("DLPFC", "Embryo") else ""
    path = os.path.join(ROOT, "generate_data", prefix, dataset_id, "data.h5ad")
    adata = sc.read_h5ad(path)

    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground'].values
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']

    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    # 返回顺序: (空间图, 特征图), 与 SemST.forward(x, sadj, fadj, ...) 保持一致
    return adata, features, labels, nsadj, nfadj


def build_neighbors(sadj):
    """从稀疏邻接矩阵构建 {spot -> set(邻居)} 映射"""
    coo = sadj.tocoo()
    neigh = defaultdict(set)
    for i, j in zip(coo.row, coo.col):
        if i != j:
            neigh[i].add(j)
    return neigh


def analyze(dataset_id, out_dir=None):
    config_name = resolve_config(dataset_id)
    config = Config(os.path.join(ROOT, "config", f"{config_name}.ini"))

    out_dir = out_dir or os.path.join(ROOT, "error_analysis", "output", dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    adata, features, labels, nsadj, nfadj = load_data(dataset_id, config_name)
    N = adata.n_obs
    class_num = int(len(np.unique(labels)))
    gt_name = adata.obs['ground_truth'].values.astype(str)

    llm_emb, llm_dim = run_precomputation_llm_base(
        os.path.join(ROOT, "data", "npys", f"embeddings_{dataset_id}.npy"))

    model = SemST(nfeat=config.fdim, nhid1=config.nhid1, nhid2=config.nhid2,
                  dropout=config.dropout, llm_dim=llm_dim,
                  llm_modulation_ratio=config.llm_modulation_ratio, use_llm=True)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "data", "weights", f"model_{dataset_id}.pth"),
        map_location='cpu'))
    model.eval()
    with torch.no_grad():
        emb, _, _, _, _, _ = model(features, nsadj, nfadj, llm_emb)
    emb = pd.DataFrame(emb.numpy()).fillna(0).values

    kmeans = KMeans(n_clusters=class_num, n_init='auto', random_state=100).fit(emb)
    idx = kmeans.labels_
    idx_aligned = hungarian_match(labels, idx)

    ari = metrics.adjusted_rand_score(labels, idx) * 100
    nmi = metrics.normalized_mutual_info_score(labels, idx) * 100
    ami = metrics.adjusted_mutual_info_score(labels, idx) * 100
    acc = metrics.accuracy_score(labels, idx_aligned) * 100
    f1w = metrics.f1_score(labels, idx_aligned, average='weighted') * 100
    f1m = metrics.f1_score(labels, idx_aligned, average='macro') * 100

    correct = (idx_aligned == labels)

    # ---- 逐类准确率 ----
    df = pd.DataFrame({'gt': labels, 'pred': idx_aligned,
                       'correct': correct, 'gt_name': gt_name})
    grp = (df.groupby('gt')
             .agg(n=('correct', 'size'), correct=('correct', 'sum'), gt_name=('gt_name', 'first')))
    grp['acc'] = grp['correct'] / grp['n'] * 100
    grp = grp.sort_values('acc')
    grp.to_csv(os.path.join(out_dir, 'per_class_accuracy.csv'))

    # ---- 误差归因 ----
    lines = []
    lines.append(f"dataset        = {dataset_id}")
    lines.append(f"config         = {config_name}")
    lines.append(f"spots          = {N}")
    lines.append(f"classes        = {class_num}")
    lines.append(f"fdim / llm_dim = {config.fdim} / {llm_dim}")
    lines.append("")
    lines.append(f"ARI = {ari:.2f}")
    lines.append(f"NMI = {nmi:.2f}")
    lines.append(f"AMI = {ami:.2f}")
    lines.append(f"ACC = {acc:.2f}")
    lines.append(f"F1_weighted = {f1w:.2f}")
    lines.append(f"F1_macro    = {f1m:.2f}")
    lines.append("")

    # (1) 类别规模 vs 准确率
    def _fmt_acc_mean(series):
        return f"{series.mean():.1f}%" if len(series) else "N/A(无此类)"

    size_acc_corr = grp['n'].corr(grp['acc'])
    large = grp[grp['n'] >= 50]['acc']
    small = grp[grp['n'] < 50]['acc']
    lines.append("== 误差归因 ==")
    lines.append(f"类别规模-准确率 相关系数 : {size_acc_corr:.3f}")
    lines.append(f"大类别(>=50 spot)平均准确率 : {_fmt_acc_mean(large)}")
    lines.append(f"小类别(<50 spot) 平均准确率 : {_fmt_acc_mean(small)}")
    lines.append("")

    # (2) 空间邻接混淆: 被误分的点, 其预测类别是否出现在其空间邻居里
    neigh = build_neighbors(adata.obsm['sadj'])
    err_idx = np.where(~correct)[0]
    boundary_confusion = 0
    for i in err_idx:
        nb = neigh.get(i, set())
        if nb and idx_aligned[i] in set(labels[list(nb)]):
            boundary_confusion += 1
    if len(err_idx):
        lines.append(f"误差中, 预测类别是其空间邻居类别 的占比 : "
                     f"{boundary_confusion / len(err_idx) * 100:.1f}% "
                     f"({boundary_confusion}/{len(err_idx)})")
    else:
        lines.append("无误差样本")
    lines.append("")

    # (3) 最差类(准确率 0)被误分到哪
    zero_cls = grp[grp['acc'] == 0.0]
    if len(zero_cls):
        lines.append("== 0% 准确率的类被误分到哪 ==")
        for gt_id, row in zero_cls.iterrows():
            sub = df[df['gt'] == gt_id]
            dst = sub['pred'].map(
                lambda p: df.loc[df['gt'] == p, 'gt_name'].iloc[0]).value_counts()
            top = ", ".join(f"{k}({v})" for k, v in dst.head(3).items())
            lines.append(f"  {row['gt_name']} (n={int(row['n'])}) -> {top}")
    lines.append("")

    # (4) 最准的类(前 5)
    lines.append("== 准确率最高的 5 个类 ==")
    for _, row in grp.tail(5).iloc[::-1].iterrows():
        lines.append(f"  {row['gt_name']} (n={int(row['n'])}) -> {row['acc']:.1f}%")
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(out_dir, 'metrics_and_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(report + "\n")

    # 保存逐 spot 预测, 供后续画图/复核
    adata.obs['pred'] = idx_aligned.astype(str)
    adata.obs['correct'] = correct.astype(bool)
    adata.obsm['emb'] = emb
    adata.write(os.path.join(out_dir, 'predictions.h5ad'))

    return {
        'dataset_id': dataset_id, 'config_name': config_name,
        'N': N, 'class_num': class_num,
        'ari': ari, 'nmi': nmi, 'ami': ami, 'acc': acc, 'f1w': f1w, 'f1m': f1m,
        'size_acc_corr': size_acc_corr,
        'large_acc': float(large.mean()) if len(large) else None,
        'small_acc': float(small.mean()) if len(small) else None,
        'boundary_confusion_rate': boundary_confusion / len(err_idx) * 100 if len(err_idx) else 0.0,
        'out_dir': out_dir,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_id', type=str, default='Mouse_Brain_Anterior')
    args = ap.parse_args()
    r = analyze(args.dataset_id)
    for k, v in r.items():
        print(f"{k}: {v}")
