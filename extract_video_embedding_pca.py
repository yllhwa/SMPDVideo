import numpy as np
from sklearn.decomposition import PCA

# 读取嵌入文件
smp_video_embeddings = "smp_video_embeddings.npy"
embeddings = np.load(smp_video_embeddings)

# 打印原始维度
print(f"原始嵌入维度: {embeddings.shape}")

# 设置目标维度（可根据需要调整）
target_dim = 20

# 初始化并拟合 PCA
pca = PCA(n_components=target_dim)
embeddings_pca = pca.fit_transform(embeddings)

# 打印降维后维度
print(f"PCA降维后维度: {embeddings_pca.shape}")
np.save("smp_video_embeddings_pca.npy", embeddings_pca)