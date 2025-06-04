K-Means 是一种无监督学习算法，用于聚类任务，与之前讨论的分类任务（如 KNN、SVC）不同。
由于你之前使用了 `load_diabetes` 数据集并关注可视化（如学习曲线、置信概率分布），我们将使用 `load_diabetes` 数据集，
应用 K-Means 聚类，分析聚类效果，并可视化聚类结果和损失函数（即簇内平方和，Within-Cluster Sum of Squares, WCSS）
随簇数 $ K $ 的变化（肘部法则）。我们将使用 Matplotlib 自由布局进行可视化，保持与之前一致的风格。

### 1. K-Means 算法原理
- **定义**：K-Means 是一种无监督聚类算法，旨在将数据集划分为 $ K $ 个簇，使得每个样本到其所属簇中心的距离平方和最小化。
- **核心思想**：通过迭代优化簇中心（质心），将样本分配到最近的簇，形成紧凑且分离的簇。
- **适用场景**：数据分组、图像分割、客户分群等。

#### 1.1 核心步骤
1. **初始化**：
   - 随机选择 $ K $ 个样本作为初始簇中心（或使用 K-Means++ 优化初始化）。
2. **分配样本**：
   - 计算每个样本到所有簇中心的距离（通常为欧几里得距离）。
   - 将样本分配到最近的簇中心。
3. **更新簇中心**：
   - 对每个簇，计算所有样本的均值作为新的簇中心。
4. **迭代**：
   - 重复分配和更新步骤，直到簇中心不再变化（或达到最大迭代次数）。
5. **输出**：
   - 每个样本的簇标签和最终簇中心。

#### 1.2 数学表达
- **优化目标**：最小化簇内平方和（WCSS）：
  $
  J = \sum_{i=1}^n \sum_{k=1}^K r_{ik} || \mathbf{x}_i - \boldsymbol{\mu}_k ||^2
  $
  其中：
  - $\mathbf{x}_i$：第 $ i $ 个样本。
  - $\mu_k$：第 $ k $ 个簇中心。
  - $ r_{ik} $：若样本 $\mathbf{x}_i$ 属于簇 $ k $，则 $ r_{ik} = 1 $，否则为 0。
  - $ || \mathbf{x}_i - \mu_k ||^2 $：样本到簇中心的欧几里得距离平方。
- **簇中心更新**：
  $
  \mu_k = \frac{\sum_{i=1}^n r_{ik} \mathbf{x}_i}{\sum_{i=1}^n r_{ik}}
  $

#### 1.3 关键参数
- **n_clusters (K)**：簇数，需预先指定，可通过肘部法则或轮廓系数选择。
- **init**：初始化方法（`k-means++` 优化初始中心，减少随机性）。
- **max_iter**：最大迭代次数。
- **random_state**：控制随机种子，确保结果可复现。

### 2. K-Means 的特点
- **优点**：
  - 简单高效，适合大规模数据集。
  - 对球形簇效果好，计算复杂度为 O(n·K·I)，其中 $ n $ 是样本数，$ K $ 是簇数，$ I $ 是迭代次数。
- **缺点**：
  - 需预先指定 $ K $，对 $ K $ 选择敏感。
  - 对初始中心敏感，可能陷入局部最优（K-Means++ 缓解）。
  - 假设簇为球形，等方差，不适合复杂形状或密度不均的簇。
  - 对噪声和异常值敏感。
  - 对特征尺度敏感，需标准化。

### 3. K-Means 在 `load_diabetes` 数据集中的应用
- **数据集**：`load_diabetes` 包含 442 个样本，10 个特征（如年龄、BMI），目标是疾病进展值（此处忽略目标，仅用特征聚类）。
- **任务**：将样本聚类为 $ K $ 个簇，探索潜在的患者群体（如健康状况分组）。
- **与分类任务的区别**：
  - 之前讨论的 KNN、SVC、朴素贝叶斯是监督学习，使用目标值（转为二分类）。
  - K-Means 是无监督学习，仅基于特征聚类，无需目标值。
- **样本均衡化**：K-Means 不直接处理类别不平衡，但 SMOTE 可用于预处理不平衡数据（如少数群体样本较少时）。
- **置信概率**：K-Means 不输出概率，但可用簇内距离或软聚类（如 GMM）近似。

### 4. 实现目标
- **任务**：对 `load_diabetes` 数据集应用 K-Means 聚类，标准化特征，选择合适的 $ K $。
- **可视化**：
  - 绘制肘部法则曲线（WCSS vs $ K $），帮助选择最优 $ K $。
  - 绘制二维 PCA 降维后的聚类结果，展示簇分布。
- **布局**：使用 Matplotlib 自由布局，展示肘部曲线和聚类结果。

### 5. 实现步骤
1. **数据预处理**：
   - 加载 `load_diabetes`，仅使用特征。
   - 标准化特征（K-Means 对尺度敏感）。
2. **K-Means 聚类**：
   - 尝试不同 $ K $，计算 WCSS。
   - 使用 PCA 降维到二维，展示聚类结果。
3. **可视化**：
   - 绘制肘部法则曲线，确定最优 $ K $。
   - 绘制 PCA 降维后的散点图，展示簇分布。
4. **自由布局**：将肘部曲线和聚类结果并排展示。

### 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 加载糖尿病数据集（仅使用特征）
diabetes = load_diabetes()
X = diabetes.data

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算不同 K 的 WCSS
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ 是 WCSS

# 使用 K=3 进行聚类（基于肘部法则观察）
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)

# PCA 降维到二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 创建画布
fig = plt.figure(figsize=(12, 6))

# 自由布局：肘部法则曲线
ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
ax1.plot(k_range, wcss, 'b-', marker='o')
ax1.set_xlabel('簇数 K')
ax1.set_ylabel('簇内平方和 (WCSS)')
ax1.set_title('肘部法则选择 K')
ax1.grid(True, ls="--")

# 自由布局：PCA 聚类结果
ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='簇中心')
ax2.set_xlabel('PCA 第 1 主成分')
ax2.set_ylabel('PCA 第 2 主成分')
ax2.set_title('K-Means 聚类结果 (K=3)')
ax2.legend()
ax2.grid(True, ls="--")
plt.colorbar(scatter, ax=ax2, label='簇标签')

plt.show()
```

### 代码说明
- **数据预处理**：
  - 使用 `load_diabetes` 加载数据集，仅使用特征（10 维）。
  - 使用 `StandardScaler` 标准化特征，确保 K-Means 不受尺度影响。
- **K-Means 聚类**：
  - 尝试 $ K = 1 $ 到 $ 10 $，计算 WCSS（`inertia_`）。
  - 使用 `init='k-means++'` 优化初始中心，选择 $ K=3 $（基于肘部法则观察）。
- **PCA 降维**：
  - 使用 `PCA` 将数据降到二维，便于可视化聚类结果。
  - 保留簇中心在 PCA 空间的投影。
- **可视化**：
  - 使用自由布局 (`fig.add_axes`) 绘制两个子图：
    - 左侧：肘部法则曲线（WCSS vs $ K $），帮助选择最优 $ K $。
    - 右侧：PCA 降维后的散点图，展示 $ K=3 $ 的聚类结果，簇中心用红色 X 标记。
  - 添加网格、图例和颜色条，增强可读性。

### 输出结果
- **肘部法则曲线（左侧）**：
  - 显示 WCSS 随 $ K $ 增加而下降，拐点（“肘部”）提示最优 $ K $。
  - 通常 $ K=3 $ 或 $ K=4 $ 是合理选择（需观察曲线）。
- **聚类结果（右侧）**：
  - PCA 散点图展示样本分布，不同簇用不同颜色表示。
  - 红色 X 标记簇中心，颜色条显示簇标签。
- **分析**：
  - 如果簇分离清晰，说明 K-Means 有效捕捉数据结构。
  - 如果簇重叠，可能需调整 $ K $、尝试其他聚类算法（如 GMM）或检查数据预处理。

### 补充说明
- **自由布局**：使用 `fig.add_axes` 灵活控制子图位置，适合对比展示。如果需要更规则的布局，可以改用 `plt.subplots`：
  ```python
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
  # 替换 fig.add_axes
  ```
- **选择 K**：
  - 肘部法则是一种启发式方法，也可使用轮廓系数（Silhouette Score）：
    ```python
    from sklearn.metrics import silhouette_score
    sil_scores = []
    for k in k_range[1:]:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    plt.plot(k_range[1:], sil_scores, 'b-', marker='o')
    plt.xlabel('簇数 K')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数选择 K')
    plt.grid(True)
    plt.show()
    ```
- **与分类任务的关系**：
  - 之前使用 KNN、SVC、朴素贝叶斯进行二分类（结合 SMOTE 和置信概率）。
  - K-Means 是无监督聚类，无需目标值，适合探索数据结构。
  - 如果想结合分类，可以用 K-Means 结果作为特征，输入 KNN/SVC。
- **与之前模型的对比**：
  - **KNN**：监督分类，基于邻居投票，需目标值。
  - **SVC**：监督分类，最大化边界，适合复杂边界。
  - **朴素贝叶斯**：监督分类，假设特征独立，概率输出直接。
  - **K-Means**：无监督聚类，基于距离分组，无概率输出。
- **样本均衡化**：
  - K-Means 不直接处理类别不平衡，但若数据分布不均，可先用 SMOTE 平衡样本，或用加权 K-Means。
- **改进建议**：
  - 如果想分析簇内样本特性，可以打印每个簇的特征均值：
    ```python
    for k in range(3):
        print(f"簇 {k} 特征均值:", X_scaled[labels == k].mean(axis=0))
    ```
  - 如果需要软聚类（概率输出），可改用高斯混合模型（GMM）：
    ```python
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    ```
  - 如果需要保存图形：
    ```python
    plt.savefig('kmeans_clustering.png')
    ```
