`pd.date_range` 是 Pandas 中用于生成日期范围的强大函数，常用于时间序列分析。它可以生成连续的日期序列，并支持多种参数来控制频率、起止时间等。你的问题还包括如何跳过周末，我会详细讲解 `pd.date_range` 的参数用法，并提供跳过周末的具体方法。

---

### `pd.date_range` 函数签名
```python
pandas.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, inclusive='both', **kwargs)
```

---

### 核心参数说明
1. **`start`（可选）**
   - 类型：字符串、日期时间对象
   - 描述：日期范围的起始时间。
   - 示例：`'2023-01-01'` 或 `pd.Timestamp('2023-01-01')`。

2. **`end`（可选）**
   - 类型：字符串、日期时间对象
   - 描述：日期范围的结束时间。
   - 示例：`'2023-12-31'`。

3. **`periods`（可选）**
   - 类型：整数
   - 描述：生成的时间点数量。如果指定了 `periods`，则 `end` 可省略。
   - 示例：`periods=10`（生成 10 个时间点）。

4. **`freq`（可选，默认 `'D'`）**
   - 类型：字符串或 `DateOffset` 对象
   - 描述：时间间隔（频率），决定每个时间点之间的步长。
   - 常用值：
     - `'D'`：每天
     - `'H'`：每小时
     - `'M'`：每月末
     - `'B'`：每个工作日（跳过周末）
     - `'W'`：每周（默认周日）
     - `'W-MON'`：每周一
   - 示例：`freq='B'`（工作日）。

5. **`tz`（可选）**
   - 类型：字符串或 `pytz.timezone`
   - 描述：时区，例如 `'UTC'`、`'Asia/Shanghai'`。
   - 示例：`tz='America/New_York'`。

6. **`normalize`（可选，默认 `False`）**
   - 类型：布尔值
   - 描述：是否将 `start` 和 `end` 标准化为午夜 00:00。
   - 示例：`normalize=True`。

7. **`name`（可选）**
   - 类型：字符串
   - 描述：生成的时间索引的名称。
   - 示例：`name='dates'`。

8. **`inclusive`（可选，默认 `'both'`）**
   - 类型：字符串（`'both'`、`'neither'`、`'left'`、`'right'`）
   - 描述：控制是否包含边界。
   - 示例：`inclusive='left'`（只包含起始点，不包含结束点）。

---

### 基本用法示例

#### 示例 1：生成每日日期
```python
import pandas as pd

dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
print(dates)
```

**输出**：
```
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
               '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
               '2023-01-09', '2023-01-10'],
              dtype='datetime64[ns]', freq='D')
```

#### 示例 2：指定周期数
```python
dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
print(dates)
```

**输出**：
```
DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
               '2023-01-05'],
              dtype='datetime64[ns]', freq='D')
```

#### 示例 3：每小时频率
```python
dates = pd.date_range(start='2023-01-01 09:00', end='2023-01-01 12:00', freq='H')
print(dates)
```

**输出**：
```
DatetimeIndex(['2023-01-01 09:00:00', '2023-01-01 10:00:00',
               '2023-01-01 11:00:00', '2023-01-01 12:00:00'],
              dtype='datetime64[ns]', freq='H')
```

---

### 如何跳过周末
Pandas 提供了内置的工作日频率 `'B'`，可以直接跳过周末（周六和周日）。此外，还可以通过自定义过滤进一步排除特定日期（如节假日）。

#### 方法 1：使用 `freq='B'`
```python
import pandas as pd

dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
print(dates)
```

**输出**：
```
DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
               '2023-01-06', '2023-01-09', '2023-01-10'],
              dtype='datetime64[ns]', freq='B')
```

- **说明**：
  - `'B'` 表示“营业日”（Business Day），自动跳过周六（例如 2023-01-07）和周日（例如 2023-01-08）。
  - 2023-01-01 是周日，也被排除。

#### 方法 2：结合 `weekday` 过滤
如果你需要更灵活的控制（例如只保留周一到周五），可以用 `.weekday` 属性过滤：
```python
import pandas as pd

# 生成完整日期范围
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')

# 过滤掉周末（weekday: 0=周一, 5=周六, 6=周日）
workdays = dates[dates.weekday < 5]
print(workdays)
```

**输出**：
```
DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
               '2023-01-06', '2023-01-09', '2023-01-10'],
              dtype='datetime64[ns]', freq=None)
```

- **说明**：
  - `dates.weekday` 返回 0（周一）到 6（周日）的整数。
  - `< 5` 保留周一到周五。

#### 方法 3：排除节假日
如果除了周末还想跳过特定节假日，可以用 Pandas 的 `CustomBusinessDay`：
```python
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

# 定义节假日
holidays = ['2023-01-02']  # 假设这是节假日

# 创建自定义工作日频率
custom_bday = CustomBusinessDay(holidays=holidays)

dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq=custom_bday)
print(dates)
```

**输出**：
```
DatetimeIndex(['2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06',
               '2023-01-09', '2023-01-10'],
              dtype='datetime64[ns]', freq='C')
```

- **说明**：
  - `CustomBusinessDay` 默认跳过周末，再根据 `holidays` 排除指定日期。

---

### 注意事项
1. **参数组合**：
   - 必须提供 `start` 和 `end` 或 `start` 和 `periods`，否则会报错。
   - 示例：`pd.date_range(start='2023-01-01')` 会失败，需指定 `end` 或 `periods`。

2. **频率选项**：
   - `'B'` 只考虑周末，不包括法定节假日。
   - 用 `CustomBusinessDay` 或外部节假日库（如 `holidays`）处理节假日。

3. **时区**：
   - 如果指定 `tz`，生成的日期会带有时区信息：
     ```python
     dates = pd.date_range(start='2023-01-01', periods=3, freq='D', tz='UTC')
     print(dates)
     # 输出：
     # DatetimeIndex(['2023-01-01 00:00:00+00:00', '2023-01-02 00:00:00+00:00',
     #                '2023-01-03 00:00:00+00:00'],
     #               dtype='datetime64[ns, UTC]', freq='D')
     ```

---

### 总结
- **`pd.date_range` 用法**：
  - 通过 `start`、`end`、`periods` 和 `freq` 生成日期序列。
  - 常用频率：`'D'`（每天）、`'B'`（工作日）、`'H'`（每小时）。
- **跳过周末**：
  - 用 `freq='B'`：简单直接，自动排除周六和周日。
  - 用 `weekday` 过滤：更灵活，适用于自定义需求。
  - 用 `CustomBusinessDay`：可同时跳过周末和节假日。
