import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scanpy as sc
import os

"""计算每个类别的准确率"""

current_dir = os.path.dirname(os.path.abspath(__file__))
adata = sc.read_h5ad(os.path.join(current_dir, '_tmp_analysis.h5ad'))

labels = adata.obs['ground'].values
pred = adata.obs['pred'].values.astype(int)
gt_name = adata.obs['gt_name'].values
correct = adata.obs['correct'].values

# per-class accuracy
df = pd.DataFrame({'gt': labels, 'pred': pred, 'correct': correct, 'gt_name': gt_name})
cls = df.groupby('gt').agg(n=('correct','size'), correct=('correct','sum'))
cls['acc'] = cls['correct']/cls['n']*100
# map class number to name
name_map = df.drop_duplicates('gt').set_index('gt')['gt_name']
cls['name'] = cls.index.map(name_map)
cls = cls.sort_values('acc')
pd.set_option('display.max_rows', 200)
print('=== per-class accuracy (worst first) ===')
print(cls[['name','n','acc']].to_string())
print()
print('total correct:', correct.sum(), 'of', len(correct), '= %.2f%%'%(correct.mean()*100))