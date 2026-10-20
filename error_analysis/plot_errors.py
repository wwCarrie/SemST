"""
生成误差可视化: 读取 error_analysis/output/<dataset>/predictions.h5ad

对每个数据集输出:
  - spatial_errors.png    左图=真实标签, 右图=正确/错误 spot 的空间分布
  - confusion_matrix.png  行归一化混淆矩阵(类别数 > 25 时跳过, 如 MBA 52 类)

用法:
    python error_analysis/plot_errors.py                 # 画全部数据集
    python error_analysis/plot_errors.py --dataset_id MVC # 只画一个
"""
from __future__ import division, print_function

import os
import sys
import warnings

import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "error_analysis", "output")

DATASETS = ['151508', 'E1S1', 'Human_Breast_Cancer', 'Mouse_Brain_Anterior', 'MVC']
# 有组织切片图像的数据集, 空间坐标按 scanpy 惯例翻转 y 轴
VISIUM = {'151508', 'Human_Breast_Cancer', 'Mouse_Brain_Anterior'}


def load(dataset_id):
    return sc.read_h5ad(os.path.join(OUT, dataset_id, "predictions.h5ad"))


def spatial_errors(adata, dataset_id, savepath):
    coords = adata.obsm['spatial']
    gt = adata.obs['ground'].values
    correct = adata.obs['correct'].values.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 左: 真实标签
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], c=gt, cmap='tab20', s=6, alpha=0.85, linewidths=0)
    ax.set_title(f"{dataset_id}: ground truth ({len(np.unique(gt))} classes)")
    ax.set_aspect('equal')
    if dataset_id in VISIUM:
        ax.invert_yaxis()

    # 右: 正确 / 错误
    ax = axes[1]
    ax.scatter(coords[~correct, 0], coords[~correct, 1], c='#d62728', s=6,
               label='error', alpha=0.9, linewidths=0)
    ax.scatter(coords[correct, 0], coords[correct, 1], c='#bcbcbc', s=6,
               label='correct', alpha=0.9, linewidths=0)
    ax.set_title(f"{dataset_id}: correct = {correct.mean() * 100:.1f}%")
    ax.legend(markerscale=3)
    ax.set_aspect('equal')
    if dataset_id in VISIUM:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def confusion_plot(adata, dataset_id, savepath):
    gt = adata.obs['ground'].values
    pred = adata.obs['pred'].values.astype(int)
    k = len(np.unique(gt))
    if k > 25:
        print(f"  [skip confusion matrix] {dataset_id}: {k} classes too many")
        return False

    cm = confusion_matrix(gt, pred, labels=list(range(k))).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    names = []
    for c in range(k):
        n = adata.obs.loc[adata.obs['ground'] == c, 'ground_truth'].iloc[0]
        names.append(f"{c}:{n}")

    fig, ax = plt.subplots(figsize=(max(6.5, k * 0.55), max(5.5, k * 0.5)))
    im = ax.imshow(cm, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{dataset_id}: row-normalized confusion (recall)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def plot_one(dataset_id):
    adata = load(dataset_id)
    d = os.path.join(OUT, dataset_id)
    spatial_errors(adata, dataset_id, os.path.join(d, "spatial_errors.png"))
    print(f"  saved spatial_errors.png")
    if confusion_plot(adata, dataset_id, os.path.join(d, "confusion_matrix.png")):
        print(f"  saved confusion_matrix.png")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_id', type=str, default=None)
    args = ap.parse_args()

    targets = [args.dataset_id] if args.dataset_id else DATASETS
    for ds in targets:
        print(f"[{ds}]")
        plot_one(ds)
