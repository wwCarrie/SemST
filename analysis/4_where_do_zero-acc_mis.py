import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scanpy as sc
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
adata = sc.read_h5ad(os.path.join(current_dir, '_tmp_analysis.h5ad'))

"""针对准确率为最低的，分析预测成的标签名"""

labels = adata.obs['ground'].values
pred = adata.obs['pred'].values.astype(int)
gt_name = adata.obs['gt_name'].values
correct = adata.obs['correct'].values
name_of = lambda idx: gt_name[idx]
df = pd.DataFrame({'gt':labels,'pred':pred,'gt_name':gt_name})
# for the 0%-accuracy classes, where did their spots go?
zero = ['MOB::MI','ORB::L5','AOB::Ml','Py','ORB::L6','FRP::L2/3']
for z in zero:
    sub = df[df.gt_name==z]
    if len(sub)==0: continue
    dst = sub.pred.map(name_of).value_counts()
    print(f'{z} (n={len(sub)}) -> mostly predicted as: {dict(dst.head(3))}')