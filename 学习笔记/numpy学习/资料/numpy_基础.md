## Numpy

在NumPy中，元数据（metadata）通常不是以某种特定的“元数据格式”单独存储，而是通过数组对象的属性和结构来体现。NumPy数组（`numpy.ndarray`）本身包含了一些关键的元数据，这些信息描述了数组的特性。以下是NumPy元数据的核心组成部分及其格式的说明：

### 1. **形状（Shape）**
   - 通过属性 `.shape` 表示，是一个元组（tuple），描述数组的每个维度的大小。
   - 示例：对于一个 3x4 的二维数组，`.shape` 返回 `(3, 4)`。

### 2. **数据类型（Dtype）**
   - 通过属性 `.dtype` 表示，定义了数组中每个元素的数据类型（如 `int32`, `float64`, `uint8` 等）。
   - NumPy 支持多种内置数据类型，还可以通过自定义 `dtype` 来描述复杂的结构化数据。
   - 示例：`np.array([1, 2, 3]).dtype` 可能返回 `int64`。

### 3. **维度（Ndim）**
   - 通过属性 `.ndim` 表示，是一个整数，表示数组的维度数量。
   - 示例：一个标量数组的 `.ndim` 为 0，一维数组为 1，二维数组为 2。

### 4. **大小（Size）**
   - 通过属性 `.size` 表示，是数组中元素的总数（所有维度大小的乘积）。
   - 示例：对于形状为 `(3, 4)` 的数组，`.size` 返回 12。

### 5. **步幅（Strides）**
   - 通过属性 `.strides` 表示，是一个元组，描述在内存中沿每个维度移动到下一个元素所需的字节数。
   - 这与数组的内存布局（C顺序或Fortran顺序）密切相关。
   - 示例：对于一个 `(3, 4)` 的 `int32` 数组（每个元素 4 字节），C顺序的 `.strides` 可能是 `(16, 4)`。

### 6. **内存布局（Flags）**
   - 通过属性 `.flags` 查看，包含关于数组内存布局的信息，比如是否是 C 连续（`C_CONTIGUOUS`）、Fortran 连续（`F_CONTIGUOUS`）或只读（`WRITEABLE`）等。

### 7. **数据缓冲区（Data Buffer）**
   - 实际数据存储在底层的缓冲区中，可以通过 `.data` 属性访问（返回一个 `memoryview` 对象）。
   - 这部分不是传统意义上的元数据，而是数组内容的载体。

### 示例代码
```python
import numpy as np

# 创建一个简单的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 查看元数据
print("Shape:", arr.shape)        # (2, 3)
print("Dtype:", arr.dtype)       # int64（具体取决于系统）
print("Ndim:", arr.ndim)         # 2
print("Size:", arr.size)         # 6
print("Strides:", arr.strides)   # (24, 8)（假设 int64，每个元素 8 字节）
print("Flags:", arr.flags)       # 显示内存布局信息
```

### 文件中的元数据（`.npy` 格式）
如果你指的是NumPy保存到磁盘上的文件格式（如 `.npy` 文件），它的元数据存储在文件头部：
- **头部结构**：以文本形式存储，包含一个 Python 字典，描述数组的 `shape`、`dtype` 和 `order`（C 或 Fortran）。
- **示例头部**：`{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4)}`。
- 头部后面跟着二进制数据。

可以用 `np.load()` 加载 `.npy` 文件时查看这些元数据：
```python
arr = np.load("example.npy")
print(arr.shape, arr.dtype)
```

### 总结
NumPy 的“元数据”主要通过数组对象的属性（如 `shape`、`dtype` 等）体现，而在文件存储时（如 `.npy`），则以头部字典的形式记录。
