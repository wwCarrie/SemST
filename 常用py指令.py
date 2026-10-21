
"""h5ad空间转录组数据文件 AnnData 格式
通常包含表达矩阵、细胞/spot 信息、基因信息等
"""

import scanpy as sc

# 读取文件
adata = sc.read_h5ad("your_data.h5ad")

# 1. 查看数据基本信息
print(adata)

# 2. 查看 spot 数量和基因数量
print("Spot 数量：", adata.n_obs)
print("基因数量：", adata.n_vars)

# 3. 查看前 5 个 spot 的信息
print(adata.obs.head())

# 4. 查看前 10 个基因
print(adata.var.head(10))

# 5. 查看有哪些空间坐标、嵌入数据
print(adata.obsm.keys())

# 6. 查看表达矩阵前 5 行、前 10 列
print(adata[:5, :10].X.toarray()
      if hasattr(adata[:5, :10].X, "toarray")
      else adata[:5, :10].X)


"""h5特征稀疏矩阵
主要存储表达矩阵和相关特征信息
"""

import scanpy as sc

adata = sc.read_10x_h5("你的文件路径.h5")

# spot 数量、基因数量
print("Spot 数量：", adata.n_obs)
print("基因数量：", adata.n_vars)

# 查看 spot 的条形码
print(adata.obs.head())

# 查看基因信息
print(adata.var.head())

# 查看表达矩阵
print(adata.X.shape)