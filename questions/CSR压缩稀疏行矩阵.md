# BGE-M3 稀疏向量提取全流程解析

## 1. 核心前置知识

BGE-M3 模型生成的稠密向量（Dense）是普通的列表，但生成的稀疏向量（Sparse）为了节省内存，底层使用的是 SciPy 库的 **CSR（Compressed Sparse Row，压缩稀疏行）矩阵格式**。

而 Milvus 等向量数据库在存储稀疏向量时，通常要求格式为 `Dict[int, float]`（即：`{Token ID: 权重}`）。

**代码的本质**：将底层的 CSR 矩阵“翻译”成数据库能看懂的字典格式。

---

## 2. 全流程代码示例

假设输入两句话：`["我喜欢苹果手机", "我是一名电子产品爱好者"]`

### 2.1 生成嵌入向量

```python
# 1. 定义输入
texts = ["我喜欢苹果手机", "我是一名电子产品爱好者"]

# 2. 生成嵌入
embeddings = bge_m3.encode_documents(texts)
```

*   **解析**：模型一次性输出字典 `embeddings`：
    *   `embeddings["dense"]`：包含 2 个稠密向量的列表（每个长度为 1024）。
    *   `embeddings["sparse"]`：一个形状为 `(2, VocabSize)` 的 CSR 稀疏矩阵。

### 2.2 提取稠密向量

```python
# 3. 提取稠密向量
dense_vectors = embeddings["dense"].tolist()   
```

*   **结果**：`[[0.1, -0.2, ...], [0.3, 0.05, ...]]`

### 2.3 从 CSR 矩阵提取稀疏向量（核心）

```python
# 4. 从 CSR 矩阵提取稀疏向量
sparse_matrix = embeddings["sparse"]
sparse_vectors = []

# 遍历每一句话
for i in range(len(texts)): 
    # 获取第 i 句话非零元素的起止索引
    start_idx = sparse_matrix.indptr[i]
    end_idx = sparse_matrix.indptr[i + 1]
    
    # 提取对应的 Token IDs 和 权重
    token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
    weights = sparse_matrix.data[start_idx:end_idx].tolist()
    
    # 打包成字典
    sparse_vector = dict(zip(token_ids, weights))
    sparse_vectors.append(sparse_vector)
```

---

## 3. CSR 矩阵底层原理图解

CSR 矩阵将“长得很胖的表格”压缩成 **3 个一维数组**。

### 3.1 原始大矩阵（未压缩状态）

假设词表维度为 6，实际数据如下：

| 行号(句子) | 词0 | 词1 | 词2 | 词3 | 词4 | 词5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **0.8** | 0 | **0.9** | 0 | 0 | **0.7** |
| **1** | 0 | **0.6** | 0 | **0.5** | **0.4** | 0 |

### 3.2 CSR 压缩后的三个数组

1.  **`data`（存权重）**：非零数字从左到右、从上到下排列。
    > `[0.8, 0.9, 0.7, 0.6, 0.5, 0.4]`

2.  **`indices`（存 Token ID）**：非零数字原本所在的列号。
    > `[0, 2, 5, 1, 3, 4]`
    > *（例如：0.8 在第 0 列，0.9 在第 2 列...）*

3.  **`indptr`（索引指针）**：记录每一行数据在 `data` 数组中的起止位置。
    > `[0, 3, 6]`
    > *（第 0 句：`data[0:3]`；第 1 句：`data[3:6]`）*

### 3.3 代码切片逻辑演示

**处理第 0 句话 (i=0)：**

```python
start_idx = indptr[0]  # 0
end_idx = indptr[1]    # 3

token_ids = indices[0:3]  # [0, 2, 5]
weights = data[0:3]       # [0.8, 0.9, 0.7]

# 结果：{0: 0.8, 2: 0.9, 5: 0.7}
```

**处理第 1 句话 (i=1)：**

```python
start_idx = indptr[1]  # 3
end_idx = indptr[2]    # 6

token_ids = indices[3:6]  # [1, 3, 4]
weights = data[3:6]       # [0.6, 0.5, 0.4]

# 结果：{1: 0.6, 3: 0.5, 4: 0.4}
```

---

## 4. 最终输出结果

```python
# 稠密向量列表
dense_vectors = [
    [0.1, -0.2, ...], 
    [0.3, 0.05, ...]
]

# 稀疏向量列表（可直接存入 Milvus）
sparse_vectors = [
    {0: 0.8, 2: 0.9, 5: 0.7}, 
    {1: 0.6, 3: 0.5, 4: 0.4}
]
```

> **总结**：CSR 矩阵通过 `indptr` 作为“剪刀”，根据句子序号将 `data` 和 `indices` 准确剪成一段一段，最后用 `zip` 拼成字典。这是处理 BGE-M3 稀疏向量最标准、最高效的写法。