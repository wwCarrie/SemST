import warnings; warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, torch, scanpy as sc
from sklearn.cluster import KMeans
from sklearn import metrics
from models import SemST
from preprocess.config import Config
from utils import *

"""生成分析文件"""

dataset_id = 'Mouse_Brain_Anterior'
adata = sc.read_h5ad(f'generate_data/{dataset_id}/data.h5ad')
features = torch.FloatTensor(adata.X)
labels = adata.obs['ground'].values
fadj = adata.obsm['fadj']; sadj = adata.obsm['sadj']
nfadj = normalize_sparse_matrix(fadj + __import__('scipy.sparse', fromlist=['sp']).eye(fadj.shape[0]))
nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
nsadj = normalize_sparse_matrix(sadj + __import__('scipy.sparse', fromlist=['sp']).eye(sadj.shape[0]))
nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

config = Config('config/Mouse_Brain_Anterior.ini')
precomputed_llm_base_embeddings, llm_base_emb_dim = run_precomputation_llm_base('data/npys/embeddings_Mouse_Brain_Anterior.npy')
model = SemST(nfeat=config.fdim, nhid1=config.nhid1, nhid2=config.nhid2, dropout=config.dropout, llm_dim=llm_base_emb_dim, llm_modulation_ratio=config.llm_modulation_ratio, use_llm=True)
model.load_state_dict(torch.load('data/weights/model_Mouse_Brain_Anterior.pth', map_location='cpu'))
model.eval()
with torch.no_grad():
    emb, mean, _, _, _, _ = model(features, nsadj, nfadj, precomputed_llm_base_embeddings)
emb = pd.DataFrame(emb.numpy()).fillna(0).values
kmeans = KMeans(n_clusters=config.class_num if hasattr(config,'class_num') else 52, n_init='auto', random_state=100).fit(emb)
# class_num not set here, use 52
kmeans = KMeans(n_clusters=52, n_init='auto', random_state=100).fit(emb)
idx = kmeans.labels_
idx_aligned = hungarian_match(labels, idx)
ari = metrics.adjusted_rand_score(labels, idx)*100
nmi = metrics.normalized_mutual_info_score(labels, idx)*100
acc = metrics.accuracy_score(labels, idx_aligned)*100
print('ARI %.2f NMI %.2f ACC %.2f' % (ari, nmi, acc))

# save
save_dir = os.path.dirname(os.path.abspath(__file__))
np.save(os.path.join(save_dir, '_tmp_emb.npy'), emb)
np.save(os.path.join(save_dir, '_tmp_idx.npy'), idx_aligned)
np.save(os.path.join(save_dir, '_tmp_labels.npy'), labels)
adata.obs['pred'] = idx_aligned.astype(str)
adata.obs['gt'] = labels.astype(str)
adata.obs['correct'] = (idx_aligned == labels) # 正确匹配
adata.obs['gt_name'] = adata.obs['ground_truth'].values
adata.write(os.path.join(save_dir, '_tmp_analysis.h5ad'))
print('saved:', save_dir)