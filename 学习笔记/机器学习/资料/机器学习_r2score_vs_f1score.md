R² 分数（R-squared score）和 F1 分数（F1 score）是机器学习中常用的评估指标，但它们适用于不同的任务场景，衡量模型性能的角度也不同。以下是两者的详细区别，并以表格形式对比。

### 1. R² 分数
- **定义**：R² 分数，也叫决定系数，衡量回归模型预测值与真实值之间的拟合程度。它表示模型解释的方差占总方差的比例。
- **公式**：
  $
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $
  其中：
  - $ y_i $：真实值
  - $ \hat{y}_i $：预测值
  - $ \bar{y} $：真实值的均值
  - 分子是残差平方和（SSE），分母是总平方和（SST）。
- **取值范围**：
  - $ R^2 \leq 1 $
  - $ R^2 = 1 $：完美拟合，预测值完全匹配真实值。
  - $ R^2 = 0 $：模型预测效果等同于用均值预测。
  - $ R^2 < 0 $：模型预测比均值预测还差。
- **适用场景**：回归任务（如预测房价、温度等连续值）。
- **特点**：
  - 衡量模型对数据变异的解释能力。
  - 对异常值敏感。
  - 不直接评估分类准确性。

### 2. F1 分数
- **定义**：F1 分数是精确率（Precision）和召回率（Recall）的调和平均数，用于评估分类模型的性能，特别是在类别不平衡时。
- **公式**：
  $
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $
  其中：
  - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$（真正例/预测为正例）
  - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$（真正例/实际正例）
  - TP：真正例，FP：假正例，FN：假负例。
- **取值范围**：
  - $ 0 \leq F1 \leq 1 $
  - $ F1 = 1 $：精确率和召回率均为 100%，完美分类。
  - $ F1 = 0 $：精确率或召回率为 0，分类失败。
- **适用场景**：分类任务（尤其是二分类，如垃圾邮件检测、疾病诊断），特别适合类别不平衡的数据集。
- **特点**：
  - 平衡精确率和召回率，综合评估分类效果。
  - 对类别不平衡敏感，适合评估少数类性能。
  - 不适用于回归任务。

### 3. 对比表格

| **特性**                | **R² 分数**                              | **F1 分数**                              |
|-------------------------|------------------------------------------|------------------------------------------|
| **适用任务**            | 回归任务（预测连续值）                   | 分类任务（预测离散类别）                 |
| **定义**                | 衡量模型解释的数据方差比例               | 精确率和召回率的调和平均数               |
| **公式**                | $ R^2 = 1 - \frac{\text{SSE}}{\text{SST}} $ | $ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $ |
| **取值范围**            | $ \leq 1 $（通常 0 到 1，负值可能）   | $ 0 \leq F1 \leq 1 $                  |
| **完美值**              | $ R^2 = 1 $（完美拟合）                | $ F1 = 1 $（精确率和召回率均为 1）     |
| **零值含义**            | $ R^2 = 0 $（等同均值预测）            | $ F1 = 0 $（精确率或召回率为 0）       |
| **负值**                | 可能，模型比均值预测差                   | 无负值                                   |
| **敏感性**              | 对异常值敏感                             | 对类别不平衡敏感                         |
| **典型场景**            | 预测房价、温度、销量等                   | 垃圾邮件检测、疾病诊断、欺诈检测等       |
| **局限性**              | 不适合分类任务；对异常值敏感             | 不适合回归任务；需结合混淆矩阵分析       |

### 4. 代码示例（结合 Matplotlib 可视化）
以下是一个简单的示例，展示如何在回归任务中计算 R² 分数，在分类任务中计算 F1 分数，并使用 Matplotlib 可视化结果。结合你之前的自由布局问题，采用自由布局展示。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, f1_score
from sklearn.datasets import make_regression, make_classification

# 生成回归数据
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
y_reg_pred = reg_model.predict(X_reg)
r2 = r2_score(y_reg, y_reg_pred)

# 生成分类数据
X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
clf_model = LogisticRegression()
clf_model.fit(X_clf, y_clf)
y_clf_pred = clf_model.predict(X_clf)
f1 = f1_score(y_clf, y_clf_pred)

# 创建画布
fig = plt.figure(figsize=(10, 4))

# 自由布局：回归任务（R² 分数）
ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
ax1.scatter(X_reg, y_reg, color='blue', alpha=0.5, label='真实值')
ax1.plot(X_reg, y_reg_pred, color='red', label='预测值')
ax1.set_title(f'回归任务 (R² = {r2:.2f})')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.legend()

# 自由布局：分类任务（F1 分数）
ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
ax2.scatter(X_clf[y_clf == 0][:, 0], X_clf[y_clf == 0][:, 1], color='blue', alpha=0.5, label='类别 0')
ax2.scatter(X_clf[y_clf == 1][:, 0], X_clf[y_clf == 1][:, 1], color='red', alpha=0.5, label='类别 1')
ax2.set_title(f'分类任务 (F1 = {f1:.2f})')
ax2.set_xlabel('特征 1')
ax2.set_ylabel('特征 2')
ax2.legend()

plt.show()
```

### 5. 示例说明
- **回归部分**：使用线性回归模型，绘制真实值与预测值的散点图和拟合线，显示 R² 分数。
- **分类部分**：使用逻辑回归模型，绘制两个类别的散点图，显示 F1 分数。
- **自由布局**：通过 `fig.add_axes` 手动设置子图位置，清晰展示回归和分类结果。

### 6. 总结
- **R² 分数**：适用于回归任务，衡量模型对连续数据的拟合能力，值越接近 1 越好，但对异常值敏感。
- **F1 分数**：适用于分类任务，平衡精确率和召回率，适合类别不平衡场景，值越接近 1 越好。
- **选择依据**：
  - 如果任务是预测连续值（如房价），用 R² 分数。
  - 如果任务是分类（如疾病诊断），用 F1 分数，尤其当类别不平衡时。
- **可视化建议**：结合 Matplotlib 的自由布局（如 `add_axes`）或网格布局（如 `subplots`），可以直观比较模型性能。

如果你有具体数据集或想进一步比较其他指标（如 MSE、准确率），请提供详情，我可以为你定制代码或优化可视化！