## timedelta

`timedelta64` 是 NumPy 中用于表示**时间差**（时间间隔）的数据类型，而不是表示某个具体的时间点。它属于 NumPy 的时间相关数据类型家族，与 `datetime64`（表示具体日期和时间）一起使用，广泛应用于时间序列计算。

以下是对 `timedelta64` 时间格式的详细说明：

---

### 1. **基本定义**
- `timedelta64` 表示两个时间点之间的差值，可以是正值（未来时间减去过去时间）或负值（过去时间减去未来时间）。
- 它的值由两部分组成：
  - **数值**：表示时间差的大小。
  - **单位**：指定时间差的计量单位（如秒、分钟、天等）。

---

### 2. **时间单位**
`timedelta64` 的时间格式通过单位来定义，单位在 `dtype` 中以 `[unit]` 的形式指定。支持的单位包括：
- `Y`：年
- `M`：月
- `W`：周
- `D`：天
- `h`：小时
- `m`：分钟
- `s`：秒
- `ms`：毫秒
- `us`：微秒（μs）
- `ns`：纳秒
- `ps`：皮秒
- `fs`：飞秒
- `as`：阿秒

例如：
- `timedelta64[D]` 表示时间差以天为单位。
- `timedelta64[ms]` 表示时间差以毫秒为单位。

---

### 3. **创建方式**
可以通过以下方式创建 `timedelta64` 对象：
- **直接指定**：
  ```python
  import numpy as np
  td = np.timedelta64(1, 'D')  # 1 天
  print(td)  # 1 days
  ```
- **时间差计算**（从 `datetime64` 相减）：
  ```python
  t1 = np.datetime64('2025-03-20')
  t2 = np.datetime64('2025-03-22')
  td = t2 - t1  # 计算时间差
  print(td)  # 2 days
  print(td.dtype)  # timedelta64[ns]（默认以纳秒为单位）
  ```

---

### 4. **默认格式和精度**
- 当两个 `datetime64` 对象相减时，生成的 `timedelta64` 默认精度取决于输入的 `datetime64` 精度，通常是**纳秒（`ns`）**。
- 你可以通过显式转换来调整单位：
  ```python
  td = np.timedelta64(3600, 's')  # 3600 秒
  td_in_hours = td.astype('timedelta64[h]')  # 转换为小时
  print(td_in_hours)  # 1 hours
  ```

---

### 5. **存储和表示**
- **存储**：`timedelta64` 在内存中存储为 64 位整数，单位决定了这个整数的含义。例如，`timedelta64[D]` 的值 2 表示 2 天。
- **表示**：在输出时，NumPy 会以人类可读的形式显示（如 `2 days`、`3600 seconds`），但底层仍是一个数值和单位的组合。

---

### 6. **与 Python 的 `timedelta` 对比**
- Python 标准库中的 `datetime.timedelta` 只支持天、秒和微秒的组合，精度和单位范围较窄。
- NumPy 的 `timedelta64` 支持更广泛的单位（从阿秒到年），并且与 `datetime64` 无缝集成，适合高性能数组计算。

---

### 示例代码
```python
import numpy as np

# 创建不同单位的 timedelta64
td1 = np.timedelta64(2, 'D')    # 2 天
td2 = np.timedelta64(3600, 's') # 3600 秒
td3 = np.timedelta64(500, 'ms') # 500 毫秒

print(td1)  # 2 days
print(td2)  # 3600 seconds
print(td3)  # 500 milliseconds

# 时间差计算
t1 = np.datetime64('2025-03-20 10:00')
t2 = np.datetime64('2025-03-20 12:30')
diff = t2 - t1
print(diff)  # 9000000000000 nanoseconds (2.5小时转换为纳秒)

# 转换为其他单位
diff_in_minutes = diff.astype('timedelta64[m]')
print(diff_in_minutes)  # 150 minutes
```

---

### 7. **实际应用**
- **时间序列分析**：计算时间间隔，如每日销售额的差值。
- **数据对齐**：调整时间戳之间的偏移。
- **批量计算**：对数组中的时间差进行矢量化操作。

---

### 总结
`timedelta64` 不是传统意义上的“时间格式”（如 `YYYY-MM-DD`），而是一种**时间差的表示方式**，由数值和单位共同定义。它的灵活性在于支持多种时间单位（从阿秒到年），并且可以与 `datetime64` 配合使用。具体的“格式”取决于你指定的单位（如 `timedelta64[D]` 表示以天为单位的时间差）。