import numpy as np
import scanpy as sc

"""验证data/npys里的数据维度, 并对比原来数据形状"""

dataset_name = 'Mouse_Brain_Anterior'
emb = np.load(rf'data/npys/embeddings_{dataset_name}.npy')
adata = sc.read_h5ad(f'generate_data/{dataset_name}/data.h5ad')

print(f"shape: {emb.shape}") # adata.n_obs
print(f"dtype: {emb.dtype}") # llm_base.config.hidden_size

# 验证: 第一维是细胞数(每个细胞一个向量), 第二维是 LLM 隐藏层维度
n_cells, embed_dim = emb.shape
print(f"细胞数 = {n_cells}, 每个细胞的嵌入维度 = {embed_dim}")

print(emb.shape)          # (n_cells, dim)
print(adata.n_obs)        # 应该 == emb.shape[0]
assert emb.shape[0] == adata.n_obs, "细胞数不一致!"