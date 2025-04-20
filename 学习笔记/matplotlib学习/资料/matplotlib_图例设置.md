在 Matplotlib 中，`plt.legend()` 是一个用于在图表中添加图例（legend）的便捷函数。它可以帮助用户识别图表中不同数据系列的含义，通常与线条、散点等图形元素关联。以下通过表格和详细说明介绍 `plt.legend()` 的使用方法。

---

### 表格：`plt.legend()` 的主要参数和功能
| **参数**          | **功能**                                  | **示例**                          |
|-------------------|-------------------------------------------|-----------------------------------|
| 无参数            | 使用默认设置添加图例                      | `plt.legend()`                   |
| `labels`          | 指定图例的标签列表                        | `plt.legend(['线1', '线2'])`     |
| `handles`         | 指定图例对应的图形对象                    | `plt.legend(handles=[line1, line2])` |
| `loc`             | 设置图例位置                              | `plt.legend(loc='upper right')`  |
| `bbox_to_anchor`  | 指定图例相对于图表的精确位置（坐标）      | `plt.legend(bbox_to_anchor=(1, 1))` |
| `ncol`            | 设置图例的列数                            | `plt.legend(ncol=2)`             |
| `fontsize`        | 设置图例字体大小                          | `plt.legend(fontsize=12)`        |
| `title`           | 为图例添加标题                            | `plt.legend(title='图例标题')`   |
| `frameon`         | 是否显示图例边框                          | `plt.legend(frameon=False)`      |

---

### 详细说明

#### 1. **基本用法**
- **`plt.legend()`** 会根据绘制的图形元素（如线条、散点）自动生成图例，前提是这些元素在创建时指定了 `label` 参数。
- **示例**：
  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(0, 10, 100)
  plt.plot(x, np.sin(x), label='正弦曲线')
  plt.plot(x, np.cos(x), label='余弦曲线')
  plt.legend()  # 默认添加图例
  plt.show()
  ```
  - **效果**：图表中出现一个图例，显示“正弦曲线”和“余弦曲线”。

#### 2. **指定标签**
- 如果没有在绘图时设置 `label`，可以用 `plt.legend(labels)` 手动指定：
  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(0, 10, 100)
  plt.plot(x, np.sin(x))
  plt.plot(x, np.cos(x))
  plt.legend(['Sin', 'Cos'])  # 手动指定标签
  plt.show()
  ```

#### 3. **控制图例位置**
- **`loc`** 参数指定图例的大致位置，可选值包括：
  - `'upper left'`、`'upper right'`、`'lower left'`、`'lower right'`
  - `'center'`、`'best'`（自动选择不遮挡数据的位置）
- **示例**：
  ```python
  plt.plot(x, np.sin(x), label='Sin')
  plt.legend(loc='upper left')
  plt.show()
  ```

- **`bbox_to_anchor`** 和 `loc` 结合使用，精确控制位置：
  - 参数为 `(x, y)`，表示相对于图表的坐标（0 到 1）。
  ```python
  plt.plot(x, np.sin(x), label='Sin')
  plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # 图例在图外右侧
  plt.show()
  ```

#### 4. **使用 `handles`**
- 当需要手动指定图例对应的图形对象时，使用 `handles` 参数：
  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(0, 10, 100)
  line1, = plt.plot(x, np.sin(x))
  line2, = plt.plot(x, np.cos(x))
  plt.legend(handles=[line1, line2], labels=['Sin', 'Cos'])
  plt.show()
  ```
  - **注意**：`plot()` 返回的是一个列表，需用 `line1,` 解包。

#### 5. **多列图例**
- 使用 `ncol` 参数将图例排列为多列：
  ```python
  plt.plot(x, np.sin(x), label='Sin')
  plt.plot(x, np.cos(x), label='Cos')
  plt.plot(x, np.tan(x), label='Tan')
  plt.legend(ncol=3)  # 三列显示
  plt.show()
  ```

#### 6. **自定义样式**
- **字体大小**：`fontsize`
- **边框**：`frameon`
- **标题**：`title`
  ```python
  plt.plot(x, np.sin(x), label='Sin')
  plt.plot(x, np.cos(x), label='Cos')
  plt.legend(title='函数', fontsize=12, frameon=False)
  plt.show()
  ```
  - **效果**：无边框图例，标题为“函数”，字体大小 12。

#### 7. **与 `Axes` 对象使用**
- 如果使用面向对象接口，可以调用 `axes.legend()`，效果相同：
  ```python
  fig, ax = plt.subplots()
  ax.plot(x, np.sin(x), label='Sin')
  ax.legend(loc='best')
  plt.show()
  ```

---

### 注意事项
1. **标签缺失**：
   - 如果绘图时未设置 `label`，`plt.legend()` 不会有内容，除非手动指定 `labels`。
2. **多子图**：
   - `plt.legend()` 作用于当前活动的 `Axes`，多子图时建议用 `ax.legend()`。
3. **顺序**：
   - 图例顺序默认与绘图顺序一致，可通过调整 `handles` 和 `labels` 的顺序改变。

---

### 综合示例
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), label='正弦', color='blue')
plt.plot(x, np.cos(x), label='余弦', color='red')
plt.plot(x, np.tan(x), label='正切', color='green')

plt.legend(
    loc='upper left',
    bbox_to_anchor=(1, 1),  # 图例放在图外
    ncol=1,
    title='三角函数',
    fontsize=10,
    frameon=True
)

plt.tight_layout()  # 调整布局避免图例遮挡
plt.show()
```
- **效果**：图例在右侧，垂直排列，带有标题“三角函数”。

---

### 总结
- **`plt.legend()`** 是添加图例的快捷方法，依赖绘图时的 `label` 或手动指定的 `labels`。
- 通过 `loc`、`bbox_to_anchor`、`ncol` 等参数，可以灵活调整图例的位置和样式。
- 对于复杂图表，推荐使用 `axes.legend()` 以明确指定作用的子图。

如果你有具体的图例设置需求（比如“图例放在底部两列”），可以告诉我，我帮你写代码！