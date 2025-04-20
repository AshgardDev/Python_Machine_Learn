## numpy基础数据类型

NumPy 提供了丰富的基础数据类型（`dtype`），这些类型用于定义数组中元素的性质。它们主要基于 C 语言的数据类型，但 NumPy 对其进行了扩展和封装，使其更灵活且与 Python 无缝集成。以下是 NumPy 的基础数据类型的分类和详细说明：

---

### 1. **整数类型（Integer Types）**
NumPy 支持多种位宽的有符号和无符号整数类型：
- **有符号整数**：
  - `int8`：8 位整数（-128 到 127）
  - `int16`：16 位整数（-32,768 到 32,767）
  - `int32`：32 位整数（-2,147,483,648 到 2,147,483,647）
  - `int64`：64 位整数（-9,223,372,036,854,775,808 到 9,223,372,036,854,775,807）
- **无符号整数**：
  - `uint8`：8 位无符号整数（0 到 255）
  - `uint16`：16 位无符号整数（0 到 65,535）
  - `uint32`：32 位无符号整数（0 到 4,294,967,295）
  - `uint64`：64 位无符号整数（0 到 18,446,744,073,709,551,615）
- **默认整数**：
  - `int_`：平台相关的默认整数类型（通常是 `int64` 或 `int32`，取决于系统）。

---

### 2. **浮点数类型（Floating-Point Types）**
用于表示小数，支持不同的精度：
- `float16`：16 位半精度浮点数（较小的范围和精度）。
- `float32`：32 位单精度浮点数（约 7 位十进制精度）。
- `float64`：64 位双精度浮点数（约 15-17 位十进制精度）。
- `float128`：128 位扩展精度浮点数（依赖于平台支持，可能不可用）。
- **默认浮点数**：
  - `float_`：默认是 `float64`。

---

### 3. **复数类型（Complex Types）**
用于表示复数，包含实部和虚部：
- `complex64`：由两个 `float32` 组成（实部和虚部各 32 位）。
- `complex128`：由两个 `float64` 组成（实部和虚部各 64 位）。
- `complex256`：由两个 `float128` 组成（依赖平台支持）。
- **默认复数**：
  - `complex_`：默认是 `complex128`。

---

### 4. **布尔类型（Boolean Type）**
- `bool`：布尔值（`True` 或 `False`），占用 8 位（1 字节）。
- **注意**：与 Python 的 `bool` 类型兼容，但固定为 1 字节存储。

---

### 5. **字符串类型（String Types）**
用于存储固定长度的字符串：
- `string_` 或 `S`：固定长度的字节字符串（如 `S10` 表示长度为 10 的字节字符串）。
- **示例**：`np.array(['abc', 'def'], dtype='S3')` 会创建长度为 3 的字节字符串数组。
- **注意**：默认编码是字节形式（`bytes`），而不是 Unicode。

---

### 6. **Unicode 字符串类型（Unicode String Types）**
用于存储固定长度的 Unicode 字符串：
- `unicode_` 或 `U`：固定长度的 Unicode 字符串（如 `U10` 表示长度为 10 的 Unicode 字符串）。
- **示例**：`np.array(['你好', '世界'], dtype='U2')`。

---

### 7. **时间类型（Datetime and Timedelta Types）**
用于处理日期、时间和时间间隔：
- `datetime64`：日期时间类型，支持多种单位（如 `datetime64[ns]` 表示纳秒精度）。
- `timedelta64`：时间差类型，同样支持多种单位（如 `timedelta64[ms]` 表示毫秒）。
- **示例**：
  ```python
  np.array(['2025-03-20', '2025-03-21'], dtype='datetime64[D]')  # 按天存储
  ```

---

### 8. **对象类型（Object Type）**
- `object` 或 `O`：允许存储任意 Python 对象（灵活但性能较低）。
- **示例**：`np.array([1, "string", [1, 2, 3]], dtype=object)`。

---

### 9. **字节顺序（Byte Order）**
NumPy 数据类型可以指定字节顺序（大端或小端）：
- `<`：小端（little-endian，如 `<i4` 表示小端 32 位整数）。
- `>`：大端（big-endian，如 `>f8` 表示大端 64 位浮点数）。
- 默认通常是系统原生字节顺序。

---

### 示例代码
```python
import numpy as np

# 不同数据类型的数组
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.5], dtype=np.float64)
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex128)
bool_arr = np.array([True, False], dtype=np.bool_)
string_arr = np.array(['abc', 'def'], dtype='S3')
unicode_arr = np.array(['你好', '世界'], dtype='U2')
datetime_arr = np.array(['2025-03-20'], dtype='datetime64[D]')

# 打印类型
print(int_arr.dtype)        # int32
print(float_arr.dtype)      # float64
print(complex_arr.dtype)    # complex128
print(bool_arr.dtype)       # bool
print(string_arr.dtype)     # |S3
print(unicode_arr.dtype)    # <U2
print(datetime_arr.dtype)   # datetime64[D]
```

---

### 如何指定数据类型
- 在创建数组时通过 `dtype` 参数指定：`np.array([1, 2, 3], dtype=np.float32)`。
- 使用简写：`np.array([1, 2, 3], dtype='f4')`（`f4` 表示 `float32`）。
- 转换类型：`arr.astype(np.int16)`。

---

### 总结
NumPy 的基础数据类型非常丰富，涵盖整数、浮点数、复数、布尔值、字符串、时间等，适用于各种科学计算场景。这些类型的选择会影响内存使用和计算性能，因此在实际应用中需要根据需求权衡精度和效率。