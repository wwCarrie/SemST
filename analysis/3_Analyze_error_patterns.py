import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scanpy as sc
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
adata = sc.read_h5ad(os.path.join(current_dir, '_tmp_analysis.h5ad'))

"""验证3个假设：
    1.样本数量大小
    2.邻近位置
    3.分层
"""

labels = adata.obs['ground'].values
pred = adata.obs['pred'].values.astype(int)
gt_name = adata.obs['gt_name'].values
correct = adata.obs['correct'].values
sadj = adata.obsm['sadj'].tocoo()
spatial = adata.obsm['spatial']

# 1. 大小 class size vs accuracy correlation
df = pd.DataFrame({'gt':labels,'correct':correct})
g = df.groupby('gt').agg(n=('correct','size'), acc=('correct','mean'))
print('corr(size, acc):', g['n'].corr(g['acc']).round(3))
print('large(>=50) classes mean acc: %.1f%%'%(g[g.n>=50]['acc'].mean()*100))
print('small(<50) classes mean acc: %.1f%%'%(g[g.n<50]['acc'].mean()*100))

# 2. 相邻 error direction: is predicted class spatially adjacent to true class?
# build neighbor mapping from sadj
rows, cols = sadj.row, sadj.col
from collections import defaultdict
neigh = defaultdict(set)
for i,j in zip(rows,cols):
    neigh[i].add(j)
boundary_matches = 0; total_err = 0
same_region = 0
for i in np.where(~correct)[0]:
    total_err += 1
    # does any spatial neighbor share the PREDICTED class?
    nb = neigh[i]
    pred_nb = set(pred[list(nb)])
    if pred[i] in pred_nb:
        boundary_matches += 1
    # does predicted class appear among neighbors' ground truth?
    gt_nb = set(labels[list(nb)])
    if pred[i] in gt_nb:
        same_region += 1
print('errors where predicted class is a spatially-adjacent type: %.1f%%'%(same_region/total_err*100))

# 3. confusion within layers (transcriptionally similar): MO::L2/3 vs MO::L5 vs MO::L6 etc
# 对同一脑区不同层的分层标注点错误、分数混淆
layer_names = [n for n in gt_name if '::L' in str(n) or '::L1' in str(n)]
print('--- layer-subtype errors (spot assigned to wrong layer of same brain region) ---')
region_err = 0; total_err_l = 0
for i in np.where(~correct)[0]:
    gt = str(gt_name[i]); pr = str(gt_name[pred[i]])
    # region prefix before ::
    def region(x):
        return x.split('::')[0] if '::' in x else x
    if region(gt)==region(pr) and ('::' in gt) and ('::' in pr):
        region_err += 1
        total_err_l += 1
    elif '::' in gt:
        total_err_l += 1
print('of errors on layer-annotated spots, fraction confused to SAME brain region different layer: %.1f%%'%(region_err/total_err_l*100))