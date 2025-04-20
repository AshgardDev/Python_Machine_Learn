### Pandas 数据结构 - DataFrame

DataFrame 是 Pandas 中的另一个核心数据结构，类似于一个二维的表格或数据库中的数据表。

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。

DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

DataFrame 提供了各种功能来进行数据访问、筛选、分割、合并、重塑、聚合以及转换等操作。

DataFrame 是一个非常灵活且强大的数据结构，广泛用于数据分析、清洗、转换、可视化等任务。

## DataFrame 特点：

- 二维结构： DataFrame 是一个二维表格，可以被看作是一个 Excel 电子表格或 SQL 表，具有行和列。可以将其视为多个 Series 对象组成的字典。

- 列的数据类型： 不同的列可以包含不同的数据类型，例如整数、浮点数、字符串或 Python 对象等。

- 索引：DataFrame 可以拥有行索引和列索引，类似于 Excel 中的行号和列标。

- 大小可变：可以添加和删除列，类似于 Python 中的字典。

- 自动对齐：在进行算术运算或数据对齐操作时，DataFrame 会自动对齐索引。

- 处理缺失数据：DataFrame 可以包含缺失数据，Pandas 使用 NaN（Not a Number）来表示。

- 数据操作：支持数据切片、索引、子集分割等操作。

- 时间序列支持：DataFrame 对时间序列数据有特别的支持，可以轻松地进行时间数据的切片、索引和操作。

- 丰富的数据访问功能：通过 .loc、.iloc 和 .query() 方法，可以灵活地访问和筛选数据。

- 灵活的数据处理功能：包括数据合并、重塑、透视、分组和聚合等。

- 数据可视化：虽然 DataFrame 本身不是可视化工具，但它可以与 Matplotlib 或 Seaborn 等可视化库结合使用，进行数据可视化。

- 高效的数据输入输出：可以方便地读取和写入数据，支持多种格式，如 CSV、Excel、SQL 数据库和 HDF5 格式。

- 描述性统计：提供了一系列方法来计算描述性统计数据，如 .describe()、.mean()、.sum() 等。

- 灵活的数据对齐和集成：可以轻松地与其他 DataFrame 或 Series 对象进行合并、连接或更新操作。

- 转换功能：可以对数据集中的值进行转换，例如使用 .apply() 方法应用自定义函数。

- 滚动窗口和时间序列分析：支持对数据集进行滚动窗口统计和时间序列分析。

![1](./pandas-DataStructure.png)
![2](./df-dp.png)

以下是一个表格，列出了 Python 中 Pandas 库中常见的 DataFrame 方法及其功能描述。为了满足你的需求，我尽量全面地列出方法，并以简洁的方式说明其用途：

| **方法**            | **功能描述**                                                                 |
|---------------------|------------------------------------------------------------------------------|
| `head(n)`           | 返回前 n 行数据，默认 n=5                                                   |
| `tail(n)`           | 返回最后 n 行数据，默认 n=5                                                 |
| `info()`            | 显示 DataFrame 的结构信息（如列名、数据类型、非空值计数）                    |
| `describe()`        | 生成数值列的统计信息（如计数、均值、标准差、最小值、最大值）                |
| `shape`             | 返回 DataFrame 的形状（行数, 列数），属性而非方法                           |
| `columns`           | 返回所有列名，属性而非方法                                                 |
| `index`             | 返回索引，属性而非方法                                                     |
| `loc[]`             | 通过标签（行名或列名）访问数据                                              |
| `iloc[]`            | 通过整数位置访问数据                                                       |
| `at[]`              | 通过标签快速访问单个值                                                     |
| `iat[]`             | 通过整数位置快速访问单个值                                                 |
| `drop(labels, axis)`| 删除指定行或列（axis=0 为行，axis=1 为列）                                 |
| `dropna()`          | 删除包含缺失值（NaN）的行或列                                              |
| `fillna(value)`     | 用指定值填充缺失值（NaN）                                                  |
| `isnull()`          | 返回布尔 DataFrame，标记缺失值位置                                         |
| `notnull()`         | 返回布尔 DataFrame，标记非缺失值位置                                       |
| `groupby(by)`       | 按指定列分组，可结合聚合函数（如 mean(), sum()）使用                        |
| `merge(right, on)`  | 合并两个 DataFrame，基于指定列                                             |
| `join(other)`       | 通过索引合并两个 DataFrame                                                 |
| `concat(objs)`      | 沿指定轴（行或列）拼接多个 DataFrame，静态方法                              |
| `append()`          | (已弃用，推荐用 concat) 沿行追加数据                                       |
| `sort_values(by)`   | 按指定列排序                                                               |
| `sort_index()`      | 按索引排序                                                                 |
| `apply(func)`       | 对行或列应用自定义函数                                                     |
| `map(func)`         | 对 Series 对象应用函数（常用于单列）                                       |
| `applymap(func)`    | 对 DataFrame 的每个元素应用函数                                            |
| `pivot(index, columns, values)` | 创建透视表                                                     |
| `pivot_table()`     | 创建带聚合功能的透视表（如均值、总和等）                                   |
| `melt()`            | 将宽格式转换为长格式                                                       |
| `transpose()`       | 转置 DataFrame（行列互换）                                                 |
| `T`                 | 转置 DataFrame 的属性形式                                                  |
| `sum()`             | 计算列或行的总和                                                           |
| `mean()`            | 计算列或行的均值                                                           |
| `std()`             | 计算列或行的标准差                                                         |
| `min()`             | 返回列或行的最小值                                                         |
| `max()`             | 返回列或行的最大值                                                         |
| `count()`           | 返回列或行的非空值计数                                                     |
| `value_counts()`    | 统计某列中唯一值的出现次数（适用于 Series）                                 |
| `unique()`          | 返回某列的唯一值数组（适用于 Series）                                       |
| `nunique()`         | 返回某列唯一值的数量                                                       |
| `replace(to_replace, value)` | 替换指定值                                                |
| `rename(columns)`   | 重命名列名                                                                 |
| `astype(dtype)`     | 转换列的数据类型                                                           |
| `to_csv()`          | 将 DataFrame 保存为 CSV 文件                                               |
| `to_excel()`        | 将 DataFrame 保存为 Excel 文件                                             |
| `plot()`            | 生成简单的图表（需配合 Matplotlib）                                        |

### 说明：
1. **属性 vs 方法**：像 `shape`、`columns`、`index` 是属性，直接访问无需括号；其他为方法，需加括号调用。
2. **聚合函数**：如 `sum()`、`mean()` 等，通常与 `groupby()` 或直接对 DataFrame 使用。
3. **数据操作**：如 `loc[]`、`iloc[]` 用于定位，`drop()`、`fillna()` 用于清洗数据。
4. **高级功能**：如 `pivot()`、`melt()` 用于数据重塑，`merge()`、`join()` 用于合并。

如果你需要更详细的解释某个方法（比如用法或示例代码），或者想扩展表格内容，请告诉我！

在 Python 的 pandas 库中，`DataFrame` 是一种功能强大的二维数据结构，提供了丰富的方法和属性来支持数据操作和分析。以下是对 `DataFrame` 方法的详细分类和描述：

**1. 构造函数：**

- `pd.DataFrame(data, index, columns, dtype, copy)`：创建一个 DataFrame 对象，支持自定义数据、索引、列名和数据类型。

**2. 属性和基础数据：**

- `df.index`：返回 DataFrame 的行索引。
- `df.columns`：返回 DataFrame 的列标签。
- `df.dtypes`：返回每列的数据类型。
- `df.values`：以二维 NumPy 数组的形式返回 DataFrame 的数据部分。
- `df.axes`：返回行索引和列名的列表。
- `df.ndim`：返回 DataFrame 的维度数（始终为 2）。
- `df.size`：返回 DataFrame 中元素的总数。
- `df.shape`：返回 DataFrame 的形状（行数，列数）。
- `df.memory_usage([index, deep])`：返回每列的内存使用情况（以字节为单位）。
- `df.empty`：指示 DataFrame 是否为空。
- `df.T`：返回 DataFrame 的转置。

**3. 数据查看与基本信息：**

- `df.head([n])`：返回前 n 行数据，默认前 5 行。
- `df.tail([n])`：返回后 n 行数据，默认后 5 行。
- `df.info()`：打印 DataFrame 的简要信息，包括索引、列、数据类型和内存使用情况。
- `df.describe([percentiles, include, exclude])`：生成描述性统计信息，包括计数、均值、标准差、最小值、四分位数和最大值。

**4. 数据选取与过滤：**

- `df['column_name']`：选取名为 `column_name` 的列，返回一个 Series。
- `df[['col1', 'col2']]`：选取多个列，返回一个新的 DataFrame。
- `df.loc[row_indexer, col_indexer]`：通过标签选取数据。
- `df.iloc[row_indexer, col_indexer]`：通过位置选取数据。
- `df.at[row_label, column_label]`：访问指定行和列标签的单个值。
- `df.iat[row_position, column_position]`：访问指定行和列位置的单个值。
- `df.filter([items, like, regex, axis])`：根据指定的条件过滤行或列。
- `df.query(expr)`：使用布尔表达式查询 DataFrame 的行。

**5. 数据清洗与处理：**

- `df.dropna([axis, how, thresh, subset, inplace])`：删除包含缺失值的行或列。
- `df.fillna(value, [method, axis, inplace, limit, downcast])`：用指定的值或方法填充缺失值。
- `df.replace(to_replace, value, [inplace, limit, regex, method])`：替换 DataFrame 中的指定值。
- `df.rename([mapper, index, columns, axis, copy, inplace, level, errors])`：重命名行索引或列名。
- `df.drop(labels, [axis, index, columns, level, inplace, errors])`：删除指定的行或列。
- `df.duplicated([subset, keep])`：返回布尔 Series，指示是否存在重复行。
- `df.drop_duplicates([subset, keep, inplace])`：删除重复的行。

**6. 数据转换：**

- `df.astype(dtype, [copy, errors])`：将 DataFrame 转换为指定的数据类型。
- `df.convert_dtypes([infer_objects, convert_string, convert_integer, convert_boolean, convert_floating])`：将列转换为最佳可能的数据类型。
- `df.infer_objects([copy])`：尝试为对象列推断更具体的数据类型。
- `df.copy([deep])`：复制 DataFrame。
- `df.to_numpy([dtype, copy, na_value])`：将 DataFrame 转换为 NumPy 数组。
- `df.to_dict([orient, into])`：将 DataFrame 转换为字典。

**7. 数据统计与聚合：**

- `df.count([axis, level, numeric_only])`：计算每列的非 NA/null 值的数量。
- `df.sum([axis, skipna, level, numeric_only, min_count])`：计算每列的和。
- `df.mean([axis, skipna, level, numeric_only])`：计算每列的均值。
- `df.median([axis, skipna, level, numeric_only])`：计算每列的中位数。
- `df.min([axis, skipna, level, numeric_only])`：计算每列的最小值。
- `df.max([axis, skipna, level, numeric_only])`：计算