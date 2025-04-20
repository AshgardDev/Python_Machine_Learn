在 Matplotlib 中，`plt`（即 `matplotlib.pyplot`）本身无法直接创建 3D 坐标系，因为 3D 绘图需要使用 `mpl_toolkits.mplot3d` 模块中的 `Axes3D` 类。要获取 3D 坐标系，通常需要结合 `plt.figure()` 和 `add_subplot()` 或 `subplot()`，并指定 `projection='3d'` 参数。以下通过表格和详细说明介绍如何使用 `plt` 获取 3D 坐标系，以及基本用法。

---

### 表格：获取 3D 坐标系的主要方法
| **方法**                  | **功能**                                  | **示例**                                      |
|---------------------------|-------------------------------------------|-----------------------------------------------|
| `plt.subplot(..., projection='3d')` | 创建带有 3D 坐标系的子图                  | `ax = plt.subplot(111, projection='3d')`     |
| `fig.add_subplot(..., projection='3d')` | 在 Figure 上添加 3D 子图                  | `ax = fig.add_subplot(111, projection='3d')` |
| `plt.figure().add_axes(..., projection='3d')` | 手动添加 3D 坐标系                        | `ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')` |

---

### 详细说明

#### 1. **基本步骤**
- **导入模块**：需要导入 `matplotlib.pyplot` 和 `mpl_toolkits.mplot3d`。
- **创建 3D 坐标系**：通过 `projection='3d'` 参数生成 `Axes3D` 对象。
- **绘制 3D 图**：在 `Axes3D` 上使用方法如 `plot()`, `scatter()`, `plot_surface()` 等。

#### 2. **使用 `plt.subplot()`**
- **用法**：直接通过 `plt.subplot()` 创建 3D 子图。
- **示例**：
  ```python
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np

  # 创建 3D 坐标系
  ax = plt.subplot(111, projection='3d')

  # 生成数据
  t = np.linspace(0, 10, 100)
  x = np.sin(t)
  y = np.cos(t)
  z = t

  # 绘制 3D 曲线
  ax.plot(x, y, z, label='螺旋线')
  ax.set_xlabel('X 轴')
  ax.set_ylabel('Y 轴')
  ax.set_zlabel('Z 轴')
  ax.legend()

  plt.show()
  ```
  - **效果**：显示一个 3D 螺旋线，带有 X、Y、Z 轴标签和图例。

#### 3. **使用 `fig.add_subplot()`**
- **用法**：先创建 `Figure` 对象，再添加 3D 子图，适合多子图场景。
- **示例**：
  ```python
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # 生成散点数据
  x = np.random.rand(50)
  y = np.random.rand(50)
  z = np.random.rand(50)

  # 绘制 3D 散点图
  ax.scatter(x, y, z, color='red', label='随机点')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.legend()

  plt.show()
  ```
  - **效果**：显示 50 个随机分布的 3D 散点。

#### 4. **使用 `add_axes()`**
- **用法**：手动指定 3D 坐标系的位置和大小。
- **示例**：
  ```python
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np

  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

  # 生成表面数据
  X = np.arange(-5, 5, 0.25)
  Y = np.arange(-5, 5, 0.25)
  X, Y = np.meshgrid(X, Y)
  Z = np.sin(np.sqrt(X**2 + Y**2))

  # 绘制 3D 表面
  ax.plot_surface(X, Y, Z, cmap='viridis')
  ax.set_xlabel('X 轴')
  ax.set_ylabel('Y 轴')
  ax.set_zlabel('Z 轴')

  plt.show()
  ```
  - **效果**：显示一个 3D 表面图，使用 `viridis` 颜色映射。

#### 5. **常用 3D 绘图方法**
- **`ax.plot(x, y, z)`**：绘制 3D 折线。
- **`ax.scatter(x, y, z)`**：绘制 3D 散点。
- **`ax.plot_surface(X, Y, Z)`**：绘制 3D 表面。
- **`ax.contour(X, Y, Z)`**：绘制 3D 等高线。

#### 6. **坐标轴设置**
- **`set_xlabel()`、`set_ylabel()`、`set_zlabel()`**：设置轴标签。
- **`set_xlim()`、`set_ylim()`、`set_zlim()`**：设置轴范围。
- **示例**：
  ```python
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(0, 10)
  ```

---

### 注意事项
1. **模块导入**：
   - 必须导入 `mpl_toolkits.mplot3d`，否则 `projection='3d'` 会报错。
2. **数据维度**：
   - 折线和散点需要一维数组 `(x, y, z)`。
   - 表面图需要二维网格数据 `(X, Y, Z)`。
3. **交互性**：
   - 在 Jupyter Notebook 中，添加 `%matplotlib notebook` 可启用交互旋转。
4. **与 2D 区别**：
   - 3D 坐标系通过 `Axes3D` 对象操作，不能直接用 2D 方法如 `plt.plot()`。

---

### 综合示例
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 6))

# 子图 1：3D 散点
ax1 = fig.add_subplot(121, projection='3d')
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)
ax1.scatter(x, y, z, c='b', label='散点')
ax1.set_title('3D 散点图')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# 子图 2：3D 表面
ax2 = fig.add_subplot(122, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax2.plot_surface(X, Y, Z, cmap='plasma')
ax2.set_title('3D 表面图')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()
```
- **效果**：左右两个子图，一个显示 3D 散点，一个显示 3D 表面。

---

### 总结
- **获取 3D 坐标系**：使用 `plt.subplot()` 或 `fig.add_subplot()`，设置 `projection='3d'`。
- **绘图类型**：支持折线、散点、表面等多种 3D 图。
- **自定义**：通过 `set_xlabel()` 等调整轴标签和范围。
