
### `np.loadtxt` 的使用方法
`np.loadtxt` 是 NumPy 中用于从文本文件（如 `.txt` 或 `.csv`）加载数据的函数，适用于读取结构化的数值数据。以下是它的用法和参数说明：

#### 函数签名
```python
numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
```

#### 参数说明
1. **`fname`**（必需）：文件路径或文件对象，例如 `'stock_data.csv'`。
2. **`dtype`**（可选，默认 `float`）：指定返回数组的数据类型，可以是单一类型（如 `int`）或结构化类型（如 `{'names': [...], 'formats': [...]}`）。
3. **`delimiter`**（可选，默认 `None`）：列分隔符，例如 `','` 用于 CSV 文件。
4. **`converters`**（可选，默认 `None`）：字典，指定列的自定义转换函数，例如 `{3: lambda v: float(v)}`。
5. **`skiprows`**（可选，默认 `0`）：跳过文件开头的行数，例如 `skiprows=1` 跳过标题行。
6. **`usecols`**（可选，默认 `None`）：指定读取的列，例如 `usecols=(0, 2)`。
7. **`unpack`**（可选，默认 `False`）：若为 `True`，按列返回单独数组。
8. **`comments`**（可选，默认 `'#'`）：忽略以指定字符开头的注释行。
9. **`encoding`**（可选，默认 `'bytes'`）：文件编码，例如 `'utf-8'`。
10. **`max_rows`**（可选，默认 `None`）：读取的最大行数。

#### 返回值
返回一个 NumPy 数组（`ndarray`），形状和类型由文件内容和参数决定。

---

### 基本用法示例
#### 示例 1：简单文本文件
文件 `data.txt`：
```
1 2 3
4 5 6
```
代码：
```python
import numpy as np
data = np.loadtxt('data.txt')
print(data)
```
输出：
```
[[1. 2. 3.]
 [4. 5. 6.]]
```

#### 示例 2：带标题的 CSV 文件
文件 `data.csv`：
```
x,y,z
1,2,3
4,5,6
```
代码：
```python
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
print(data)
```
输出：
```
[[1. 2. 3.]
 [4. 5. 6.]]
```

#### 示例 3：使用 `converters`
文件 `data_with_units.txt`：
```
1 2.5m 3
4 5.6m 6
```
代码：
```python
data = np.loadtxt('data_with_units.txt', converters={1: lambda s: float(s.rstrip('m'))})
print(data)
```
输出：
```
[[1.  2.5 3. ]
 [4.  5.6 6. ]]
```
