# 知识库查询 —— 重排序节点

## 1. 任务目标

本节课将实现知识库查询流程中的 **重排序节点**（`rerank`）。

该节点使用 **Reranker 模型** 对 RRF 融合结果和网络搜索结果进行精排，并通过 **断崖检测算法** 实现动态 TopK 截断，筛选出最相关的文档用于最终答案生成。

**学完本节你将掌握：**

- Reranker 模型的工作原理（交叉编码器 vs 双塔模型）
- BGE-Reranker 模型的使用方法
- 断崖检测（Cliff Detection）动态截断算法
- 双阈值检测的设计思路（绝对差距 vs 相对差距）
- 多源文档格式统一与合并技巧

---

## 2. 核心概念扫盲

### 2.1 为什么需要重排序？

RRF 融合虽然合并了多路检索结果，但存在局限性：

```
RRF 的局限：
─────────────────────────────────────────────
1. RRF 只看排名不看语义，无法判断文档是否真正回答了问题
2. 网络搜索结果没有经过 RRF，需要和本地结果统一评估
3. RRF 输出可能混杂相关与不相关的文档，需要进一步筛选
```

**为什么网络搜索结果不参与 RRF，而是在 Rerank 阶段才加入？**

RRF 的前提是多路结果之间可以通过 `chunk_id` 去重和投票。三路本地检索（向量、HyDE、知识图谱）查的都是同一个 Milvus 切片库，同一个 `chunk_id` 可能被多路同时命中，RRF 就是靠这个"多路共识"来排序的。而网络搜索结果来自外部网页，每条都是独立的 URL，跟本地切片没有任何重叠，放进 RRF 永远只有单路命中，拿不到多路投票的加分，排名会被压到最后，等于白加。

而 Reranker 不看排名也不看 `chunk_id`，它是把问题和每篇文档拼在一起过 Transformer，逐篇独立打分。不管文档来自本地还是网络，只要内容跟问题相关，就能拿到高分。所以网络搜索结果在 Reranker 阶段加入是合适的，能和本地文档站在同一个评分体系下公平竞争。

一句话总结：**RRF 靠 chunk_id 投票，网络结果没法投；Reranker 靠语义打分，网络结果能参与。**

**重排序的作用：**
- 使用专门的相关性模型进行精排（语义级别的相关性判断）
- 统一评估所有来源的文档（本地 RRF 结果 + 网络搜索结果）
- 过滤低质量文档，提高答案生成质量

### 2.2 双塔模型 vs 交叉编码器

检索系统中有两种主流的文本匹配架构：

#### 双塔模型（Bi-Encoder）

```
    Query                Document
      │                     │
      v                     v
  ┌───────┐            ┌───────┐
  │Encoder│            │Encoder│
  └───────┘            └───────┘
      │                     │
      v                     v
  [向量 Q]              [向量 D]
      │                     │
      └──────────┬──────────┘
                 │
                 v
            相似度计算
           (余弦/点积)
```

**特点：**
- Query 和 Document 独立编码
- Document 向量可预计算存储
- 速度快，适合召回阶段
- 交互信息有限，精度一般

> **为什么叫"双塔"？** 因为 Q 和 D 各走一个独立的 Encoder，两个 Encoder 各自独立、结构对称、并排竖立，画出来形状像两座并排的塔，所以叫双塔。

#### 交叉编码器（Cross-Encoder）

```
    Query + Document（拼接输入）
              │
              v
    ┌─────────────────────┐
    │   Transformer       │
    │   (BERT/RoBERTa)    │
    │                     │
    │  Q 和 D 深度交互     │
    └─────────────────────┘
              │
              v
         相关性得分
```

**特点：**
- Query 和 Document 联合编码
- 充分捕获交互信息
- 精度高，适合精排阶段
- 速度慢，无法预计算

> **为什么叫"交叉"？** 不是因为 Q 和 D 成对输入，而是因为它们在 Transformer 内部的 Attention 层**互相交叉注意**。Q 和 D 被拼成一个序列 `[CLS] Q [SEP] D [SEP]` 一起喂进 Transformer，在每一层 Self-Attention 中，Q 的每个 token 都能 attend 到 D 的每个 token，D 也能 attend 到 Q 的每个 token，Q 和 D 的信息在每一层都在交叉流动。双塔模型中 Q 和 D 各编各的，最后才见面算相似度；交叉编码器中 Q 和 D 从第一层开始就在双向交互，所以精度更高。

> **`[CLS]` 和 `[SEP]` 是什么？** 它们是 BERT 系列模型预定义的特殊标记符。`[CLS]`（Classification）放在序列开头，它对应的输出向量被当作整个序列的"汇总表示"，Reranker 最终的相关性分数就是从这个位置算出来的。`[SEP]`（Separator）是分隔符，告诉模型"这里是两段文本的边界"，模型看到 `[SEP]` 就知道前面是 Q、后面是 D。
>
> ```
> [CLS] 怎么 测 主板 短路 [SEP] 主板 短路 用 蜂鸣 档 测量 [SEP]
>  ↑                       ↑                                ↑
>  汇总位                Q和D的分界                        序列结束
>  ↓
> 最终从这里输出相关性分数
> ```
>
> 简单理解：`[CLS]` 是"给我一个总分"的占位符，`[SEP]` 是"问题到这里结束，文档从这里开始"的分隔线。

**Reranker 模型就是交叉编码器！** 它把问题和文档拼在一起输入 Transformer，让模型充分理解两者之间的语义关系，输出一个相关性分数。

### 2.3 BGE-Reranker 模型

我们使用的 **BGE-Reranker-Large** 是智源研究院开源的中英双语重排序模型：

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    model_name_or_path="BAAI/bge-reranker-large",
    device="cuda",      # GPU 加速
    use_fp16=True       # 半精度推理
)

# 计算相关性得分
pairs = [
    ("什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"),
    ("什么是万用表？", "今天天气很好")
]
scores = reranker.compute_score(pairs)
# 输出: array([0.9234, 0.0156])  高分 = 高相关，可以是负数
```

> **注意**：`compute_score` 的输入推荐使用元组 `(str, str)` 的列表，与源码签名 `List[Tuple[str, str]]` 一致。返回的是 `numpy.ndarray`，每个元素对应一个 pair 的相关性分数，分数可以是负数（表示完全不相关）。

**模型特性：**

| 特性 | 说明 |
|------|------|
| 基座模型 | XLM-RoBERTa-Large |
| 参数量 | 560M |
| 输入长度 | 最大 512 tokens |
| 输出 | 相关性分数（越高越相关，可为负数） |
| 支持语言 | 中英双语 |

### 2.4 为什么需要动态截断？

重排序后需要决定保留多少文档。传统做法是固定 TopK，但这不够灵活：

```
固定 TopK=5 的问题：

情况 1：前 3 篇高度相关，后 2 篇噪声
得分: [0.95, 0.92, 0.88, 0.12, 0.08]
                        ↑
                  应该在这里截断，但固定 Top5 会把噪声带进去

情况 2：前 7 篇都相关
得分: [0.95, 0.91, 0.87, 0.83, 0.79, 0.75, 0.71]
                              ↑
                        固定 Top5 会丢失有价值的第 6、7 篇
```

**断崖检测的思路：** 不固定数量，而是寻找得分"断崖式下跌"的位置，在那里截断。最少保留 `min_topk` 条（保底），最多保留 `max_topk` 条（封顶）。

### 2.5 断崖检测算法

#### 核心公式

```python
# 绝对差距
abs_gap = score[i] - score[i+1]

# 相对差距
rel_gap = abs_gap / (abs(score[i]) + 1e-6)

# 满足任一条件即为断崖
if abs_gap >= gap_abs or rel_gap >= gap_ratio:
    cutoff_at = i + 1
```

#### 为什么需要两个阈值？

用一个具体场景说明。假设只用绝对阈值 `gap_abs=0.5`：

```
高分区: 8.0 → 7.2   abs_gap=0.8 >= 0.5 → 截断 ✓ 没问题

低分区: 0.8 → 0.5   abs_gap=0.3 < 0.5  → 不截断 ✗ 有问题
```

低分区这个例子，从 0.8 跌到 0.5，跌了 37.5%，质量已经明显下降了，但绝对差距只有 0.3，达不到 0.5 的阈值，断崖检测失效，不相关的文档就混进去了。

如果把绝对阈值调低到 0.3 来覆盖低分区呢：

```
低分区: 0.8 → 0.5   abs_gap=0.3 >= 0.3 → 截断 ✓ 修好了

高分区: 8.0 → 7.7   abs_gap=0.3 >= 0.3 → 截断 ✗ 误杀了
```

高分区从 8.0 到 7.7 只跌了 3.75%，两篇都是高度相关的文档，不应该截断，但绝对阈值 0.3 把它误杀了。

**本质矛盾**：绝对阈值调高了低分区漏检，调低了高分区误杀，找不到一个值能同时兼顾两个区间。加上相对阈值就解决了——高分区靠绝对阈值（大数值容易产生大差距），低分区靠相对阈值（小数值容易产生大比例），各管各的地盘，互不干扰



两个条件分别对应两种断崖场景，满足任意一个就截断：

**`abs_gap >= gap_abs`（绝对断崖）** — 捕捉高分区的大幅下跌：

```
位置3: score=8.0
位置4: score=7.2
abs_gap = 0.8 >= 0.5(gap_abs) → 触发截断

rel_gap = 0.8/8.0 = 0.10 < 0.25(gap_ratio) → 不触发
```

高分区分数基数大，即使跌了不少，除以基数后比例很小，`rel_gap` 感知不到，靠 `abs_gap` 抓。

**`rel_gap >= gap_ratio`（相对断崖）** — 捕捉低分区的比例性下跌：

```
位置3: score=0.8
位置4: score=0.5
abs_gap = 0.3 < 0.5(gap_abs) → 不触发

rel_gap = 0.3/0.8 = 0.375 >= 0.25(gap_ratio) → 触发截断
```

低分区分数基数小，绝对差距不容易大，但比例容易大，靠 `rel_gap` 抓。

**为什么用 `or` 不用 `and`：** 用 `and` 的话两个条件必须同时满足，很多明显的断崖反而检测不到。用 `or` 让两个条件互相补位，高分区靠绝对差距兜住，低分区靠相对比例兜住。



#### `abs(current_score)` 的作用

分母加 `abs` 是因为 reranker 的分数可以是负数：

```
位置3: score = -1.5
位置4: score = -1.8
abs_gap = 0.3

不加 abs: rel_gap = 0.3 / (-1.5) = -0.2  ← 负数，永远不会触发截断
加了 abs: rel_gap = 0.3 / 1.5 = 0.2      ← 正数，能正常判断
```

`1e-6` 是 epsilon 保护，防止 `score` 为 0 时除以零。

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rerank_gap_abs` | 0.5 | 绝对差值阈值 |
| `rerank_gap_ratio` | 0.25 | 相对比例阈值 |
| `rerank_min_topk` | 3 | 最少保留数量（保底） |
| `rerank_max_topk` | 10 | 最多保留数量（封顶） |

---

## 3. 重排序业务处理流程（总）

### 3.1 节点在流程中的位置

```
                    multi_search
                         │
        ┌────────────────┼────────────────┬────────────────┐
        │                │                │                │
        v                v                v                v
  search_embedding  search_hyde     query_kg        web_search
        │                │                │                │
        └────────────────┴────────────────┘                │
                         │                                  │
                         v                                  │
                        rrf  ───────────────────────────────│
                         │                                  │
                         │              ┌───────────────────┘
                         │              │
                         v              v
                    rrf_chunks    web_search_docs
                         │              │
                         └──────┬───────┘
                                │
                                v
                    ┌───────────────────────┐
                    │     ★ rerank ★        │
                    │                       │
                    │  1. 合并多源文档       │
                    │  2. Reranker 精排     │
                    │  3. 断崖检测截断       │
                    └───────────────────────┘
                                │
                                v
                          reranked_docs
                                │
                                v
                          answer_output
```

### 3.2 节点输入输出

```
┌─────────────────────────────────────────────────────────────┐
│                        RerankNode                           │
├─────────────────────────────────────────────────────────────┤
│  输入:                                                      │
│    state["rrf_chunks"]        # RRF 融合后的本地文档         │
│    state["web_search_docs"]   # 网络搜索结果                 │
│    state["rewritten_query"]   # 重写后的查询（或原始查询）    │
│                                                             │
│  输出:                                                      │
│    state["reranked_docs"]     # 重排序后的文档列表           │
│      - content: 文档内容                                    │
│      - score: 相关性得分                                    │
│      - source: 来源（local/web）                            │
│      - chunk_id: 本地文档 ID（网络文档为 None）              │
│      - title: 文档标题                                      │
│      - url: 网络文档链接                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 重排序业务处理流程（分）

### 4.1 目标

实现一个重排序节点，将 RRF 融合结果和网络搜索结果统一精排，并通过断崖检测动态筛选最相关的文档。

### 4.2 需求分析

| 需求项 | 说明 |
|--------|------|
| 多源合并 | 合并本地 RRF 结果和网络搜索结果 |
| 格式统一 | 不同来源的文档统一为相同结构（content/source/chunk_id/title/url） |
| 防御性取值 | 所有字段用 `get(key, "")` 给默认值，防止上游数据不干净 |
| 精确排序 | 使用 Reranker 模型（交叉编码器）计算语义相关性得分 |
| 动态截断 | 通过断崖检测自动确定保留数量（双阈值：绝对 + 相对） |
| 降级处理 | Reranker 失败时返回原序（score=None） |
| 来源追溯 | 保留文档来源标识（local/web） |

### 4.3 实现流程

#### 4.3.1 实现流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 获取查询文本                                        │
│   - 优先使用 rewritten_query                                │
│   - 降级使用 original_query                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 合并本地 RRF 结果                                   │
│   - 遍历 rrf_chunks                                         │
│   - 提取 content、chunk_id、title                           │
│   - 标记 source = "local"                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 合并网络搜索结果                                    │
│   - 遍历 web_search_docs                                    │
│   - 提取 snippet(优先)/content、url、title                  │
│   - 标记 source = "web"                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 构建 Query-Document 对                              │
│   - 为每个文档创建 (question, content) 元组                  │
│   - 准备输入 Reranker 模型                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Reranker 计算得分                                   │
│   - 调用 FlagReranker.compute_score()                       │
│   - 为每个文档添加 score 字段                                │
│   - 按得分降序排列                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 断崖检测动态截断                                    │
│   - 从 min_topk 位置开始检测                                │
│   - 计算绝对差距(abs_gap)和相对差距(rel_gap)                │
│   - 任一超过阈值即截断                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│ Step 7: 返回重排序结果                                      │
│   - 记录日志（处理前后数量）                                 │
│   - 返回 reranked_docs                                      │
└─────────────────────────────────────────────────────────────┘
```

#### 4.3.2 具体实现步骤

##### Step 1: 获取查询文本

从状态中获取用于重排序的查询文本：

```python
question = state.get("rewritten_query") or state.get("original_query", "")
```

**为什么优先使用 rewritten_query？**

```
原始查询: "这块主板怎么修？"
重写查询: "主板维修方法和常见故障排查步骤"

重写后的查询更完整，与文档的匹配效果更好。
```

##### Step 2-3: 合并多源文档

遍历 RRF 融合结果和网络搜索结果，统一格式后合并：

```python
def _merge_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
    """合并本地 RRF 结果和网络搜索结果为统一格式。"""
    doc_items = []

    # 本地 RRF 结果
    for doc in (state.get("rrf_chunks") or []):
        if not isinstance(doc, dict):
            continue
        content = doc.get("content", "").strip()
        if not content:
            continue
        doc_items.append(self._make_doc_item(
            content=content,
            chunk_id=doc.get("chunk_id", "").strip() or None,
            title=doc.get("title", "").strip(),
            source="local",
        ))

    # 网络搜索结果
    for doc in (state.get("web_search_docs") or []):
        if not isinstance(doc, dict):
            continue
        content = doc.get("snippet", "").strip() or doc.get("content", "").strip()
        if not content:
            continue
        doc_items.append(self._make_doc_item(
            content=content,
            title=doc.get("title", "").strip(),
            url=doc.get("url", "").strip(),
            source="web",
        ))

    self.logger.info(f"合并文档: {len(doc_items)} 篇")
    return doc_items
```

> **防御性取值**：所有字段用 `get(key, "")` 给默认空字符串，再 `.strip()` 清洗。`None`、空字符串、纯空格都能兜住，不会因为上游数据不干净而崩溃。

> **`chunk_id` 特殊处理**：`.strip()` 后如果是空字符串就 `or None` 转成 `None`，语义更明确——要么有一个有效的 ID，要么就是没有。下游用 `if chunk_id` 判断就能准确区分。

**统一文档结构：**

```python
@staticmethod
def _make_doc_item(
        content: str,
        source: str = "",
        chunk_id: str = None,
        title: str = "",
        url: str = "",
) -> Dict[str, Any]:
    """构建统一的文档结构。"""
    return {
        "content": content,
        "source": source,
        "chunk_id": chunk_id,
        "title": title,
        "url": url,
    }
```

> **字段名用 `content` 而不是 `text`**：上游 RRF 传过来的字段就叫 `content`，下游答案生成节点用的也是 `content`，中间保持一致省去转换。

> **`_make_doc_item` 不做二次防御**：调用前 `_merge_docs` 已经保证了数据干净，方法职责单一只负责组装字典。

##### Step 4-5: Reranker 计算得分

构建问题-文档对，调用 Reranker 模型计算相关性得分：

```python
def _rerank(
        self, question: str, doc_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """计算相关性得分并排序，失败时降级返回原序（score=None）。"""
    if not doc_items or not question:
        return []

    try:
        reranker = get_reranker_model()
        # 构建 (问题, 文档内容) 元组列表，与源码签名 List[Tuple[str, str]] 一致
        pairs = [(question, item["content"]) for item in doc_items]
        scores = reranker.compute_score(pairs)

        scored = [
            {**item, "score": float(s)}
            for item, s in zip(doc_items, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    except Exception as e:
        self.logger.error(f"重排序失败，降级为原序: {e}")
        return [{**item, "score": None} for item in doc_items]
```

**得分计算过程：**

```
Query: "怎么测这块主板的短路问题？"

文档 1: "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量"
        → score = 0.9156 (高度相关)

文档 2: "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路"
        → score = 0.8823 (相关)

文档 3: "今天中午去吃猪脚饭吧，这块主板外观很漂亮"
        → score = 0.0412 (不相关)

文档 4: "苹果发布新款手机，A系列芯片性能提升20%"
        → score = -0.1523 (完全不相关，负数)
```

> **为什么用元组 `(question, item["content"])` 而不是列表 `[question, item["content"]]`**：Reranker 源码签名是 `List[Tuple[str, str]]`，用元组语义上表示"固定两个元素的配对"，更贴合"问题-文档对"的含义。

> **降级处理**：Reranker 模型可能加载失败、GPU 显存不足或输入文本过长，降级后 `score=None`，系统仍可工作，只是精度下降。下游断崖检测遇到 `None` 会跳过该对比，不会报错。

##### Step 6: 断崖检测动态截断

从 min_topk 位置开始逐对检查，发现断崖立即截断：

```python
def _cliff_cutoff(
        self, ranked_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """断崖检测截断：相邻得分差距超过阈值时截断。

    从 min_topk 位置开始逐对检查相邻文档的分数差，
    发现断崖（绝对差距或相对比例超过阈值）立即截断。
    最少保留 min_topk 条，最多保留 max_topk 条。

    Args:
        ranked_docs: 按 reranker 分数降序排列的文档列表，每条包含 score 字段

    Returns:
        截断后的文档列表，质量断崖之后的低分文档被丢弃
    """
    # 1. 空文档直接返回
    if not ranked_docs:
        return []

    # 2. 计算截断范围：最多不超过文档总数，最少不超过最大值
    upper_bound = min(self.config.rerank_max_topk, len(ranked_docs))
    lower_bound = min(self.config.rerank_min_topk, upper_bound)

    # 3. 默认取最大值，遇到断崖则提前截断
    cutoff_pos = upper_bound
    for i in range(lower_bound - 1, upper_bound - 1):
        current_score = ranked_docs[i].get("score")
        next_score = ranked_docs[i + 1].get("score")

        # 3.1 分数为空时跳过（reranker 降级返回 score=None 的情况）
        if current_score is None or next_score is None:
            continue

        # 3.2 计算绝对差距和相对差距
        abs_gap = current_score - next_score
        rel_gap = abs_gap / (abs(current_score) + 1e-6)

        # 3.3 任一差距超过阈值即为断崖，立即截断
        # abs_gap 捕捉高分区的大幅下跌，rel_gap 捕捉低分区的比例性下跌
        if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
            cutoff_pos = i + 1
            self.logger.debug(
                f"断崖检测: 位置 {i + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
            )
            break

    # 4. 返回截断后的文档
    return ranked_docs[:cutoff_pos]
```

**断崖检测示例：**

```
配置: gap_abs=0.5, gap_ratio=0.25, min_topk=3, max_topk=10

排序后得分: [0.92, 0.88, 0.85, 0.04, -0.15]
位置:        [1]   [2]   [3]   [4]   [5]

检测过程（从位置 3 开始，因为 min_topk=3）:
  位置 3→4: abs_gap = 0.85 - 0.04 = 0.81 >= 0.5(gap_abs)
            → 发现断崖！截断在位置 3

结果: 保留前 3 篇文档，第 4、5 篇被丢弃
```

> **`upper_bound` 用 `min`**：防止配置的 `max_topk=10` 但只有 3 篇文档时遍历越界。

> **`lower_bound` 也用 `min`**：防止 `min_topk=3` 但只有 1 篇文档时 `range(2, 0)` 语义不对。加上后 `lower_bound` 永远 ≤ `upper_bound`，逻辑更清晰。

> **`abs_gap` 不需要 `abs`**：`ranked_docs` 按分数降序排列，`current_score` 一定 ≥ `next_score`，差值永远非负。即使两个都是负分（如 `-1.5 - (-3.2) = 1.7`），前面的也一定比后面的大。

### 4.4 代码实现

完整的节点实现代码：

```python
"""重排序节点

使用 Reranker 模型对 RRF 融合结果和网络搜索结果进行重排序，
并通过断崖检测实现动态 TopK 截断。
"""

from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.tools.reranker_utils import get_reranker_model


class RerankNode(BaseNode):
    """重排序节点。

    流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
    """

    name = "rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        question = state.get("rewritten_query") or state.get("original_query", "")

        # 1. 合并文档
        doc_items = self._merge_docs(state)

        # 2. 重排序
        scored_docs = self._rerank(question, doc_items)

        # 3. 动态 TopK 截断
        topk_docs = self._cliff_cutoff(scored_docs)
        self.logger.info(f"重排序完成: {len(doc_items)} → {len(topk_docs)}")

        return {"reranked_docs": topk_docs}

    def _merge_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式。"""
        doc_items = []

        for doc in (state.get("rrf_chunks") or []):
            if not isinstance(doc, dict):
                continue
            content = doc.get("content", "").strip()
            if not content:
                continue
            doc_items.append(self._make_doc_item(
                content=content,
                chunk_id=doc.get("chunk_id", "").strip() or None,
                title=doc.get("title", "").strip(),
                source="local",
            ))

        for doc in (state.get("web_search_docs") or []):
            if not isinstance(doc, dict):
                continue
            content = doc.get("snippet", "").strip() or doc.get("content", "").strip()
            if not content:
                continue
            doc_items.append(self._make_doc_item(
                content=content,
                title=doc.get("title", "").strip(),
                url=doc.get("url", "").strip(),
                source="web",
            ))

        self.logger.info(f"合并文档: {len(doc_items)} 篇")
        return doc_items

    @staticmethod
    def _make_doc_item(
            content: str,
            source: str = "",
            chunk_id: str = None,
            title: str = "",
            url: str = "",
    ) -> Dict[str, Any]:
        """构建统一的文档结构。"""
        return {
            "content": content,
            "source": source,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
        }

    def _rerank(
            self, question: str, doc_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """计算相关性得分并排序，失败时降级返回原序（score=None）。"""
        if not doc_items or not question:
            return []

        try:
            reranker = get_reranker_model()
            pairs = [(question, item["content"]) for item in doc_items]
            scores = reranker.compute_score(pairs)

            scored = [
                {**item, "score": float(s)}
                for item, s in zip(doc_items, scores)
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        except Exception as e:
            self.logger.error(f"重排序失败，降级为原序: {e}")
            return [{**item, "score": None} for item in doc_items]

    def _cliff_cutoff(
            self, ranked_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not ranked_docs:
            return []

        upper_bound = min(self.config.rerank_max_topk, len(ranked_docs))
        lower_bound = min(self.config.rerank_min_topk, upper_bound)

        cutoff_pos = upper_bound
        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[i].get("score")
            next_score = ranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                self.logger.debug(
                    f"断崖检测: 位置 {i + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break

        return ranked_docs[:cutoff_pos]


_node_instance = RerankNode()


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """兼容原有调用方式的入口函数。"""
    return _node_instance(state)
```

**Reranker 工具模块（reranker_utils.py）：**

```python
import os
from FlagEmbedding import FlagReranker

_reranker_model = None


def get_reranker_model():
    """获取 Reranker 模型（单例模式）。"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = FlagReranker(
            model_name_or_path=os.getenv("BGE_RERANKER_LARGE"),
            device=os.getenv("BGE_RERANKER_DEVICE"),
            use_fp16=os.getenv("BGE_RERANKER_FP16")
        )
    return _reranker_model
```

---

## 5. 测试运行

### 5.1 运行重排序节点测试

```bash
# 确保配置了 Reranker 模型路径
export BGE_RERANKER_LARGE="/path/to/bge-reranker-large"
export BGE_RERANKER_DEVICE="cuda"
export BGE_RERANKER_FP16="True"

# 运行测试
python -m knowledge.processor.query_process.nodes.rerank_node
```

### 5.2 测试代码

```python
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    result = node_rerank(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
```

### 5.3 预期输出

```
============================================================
开始测试: 重排序节点 (RerankNode)
============================================================
【输入状态】:
  查询: 怎么测这块主板的短路问题？
  本地文档: 2 篇
  网络文档: 2 篇
------------------------------------------------------------
[rerank] 合并文档: 4 篇
[rerank] 断崖检测: 位置 2, abs_gap=0.4521, rel_gap=0.5123
[rerank] 重排序完成: 4 → 2

【重排序结果】:
[1] score=0.9156 | local | 主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档...
[2] score=0.8823 | web   | 主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路...
------------------------------------------------------------
测试完成
```

### 5.4 处理前后对比

| 对比项 | 处理前 | 处理后 |
|--------|--------|--------|
| 文档数量 | 4 篇（rrf 2 + web 2） | 2 篇（断崖截断） |
| 排序依据 | RRF 位置 / 搜索顺序 | Reranker 相关性得分 |
| 文档格式 | 不统一（content/snippet） | 统一结构（content） |
| 来源标识 | 无 | source: local/web |
| 相关性 | 混杂相关与不相关 | 仅保留高相关文档 |

**数据结构变化：**

```python
# 处理前
state = {
    "rrf_chunks": [
        {"chunk_id": "local_1", "content": "主板短路..."},
        {"chunk_id": "local_2", "content": "今天中午..."},
    ],
    "web_search_docs": [
        {"url": "...", "snippet": "主板通电前..."},
        {"url": "...", "snippet": "苹果发布..."},
    ]
}

# 处理后
state = {
    ...,
    "reranked_docs": [
        {
            "content": "主板短路通常表现为...",
            "score": 0.9156,
            "source": "local",
            "chunk_id": "local_1",
            "title": "主板维修手册",
            "url": ""
        },
        {
            "content": "主板通电前先打各主供电...",
            "score": 0.8823,
            "source": "web",
            "chunk_id": None,
            "title": "短路查修指南",
            "url": "https://example.com/repair"
        }
    ]
}
```

---

## 6. 总结

### 6.1 节点功能概览

```
┌─────────────────────────────────────────────────────────────┐
│                        RerankNode                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  核心功能: 使用 Reranker 模型精排并动态截断                  │
│                                                             │
│  输入:                                                      │
│    ├── state["rrf_chunks"]        RRF 融合结果              │
│    ├── state["web_search_docs"]   网络搜索结果              │
│    └── state["rewritten_query"]   查询文本                  │
│                                                             │
│  输出:                                                      │
│    └── state["reranked_docs"]     精排后的文档列表          │
│          ├── content   文档内容                             │
│          ├── score     相关性得分                           │
│          ├── source    来源 (local/web)                     │
│          ├── chunk_id  本地文档 ID（网络文档为 None）        │
│          ├── title     文档标题                             │
│          └── url       网页链接                             │
│                                                             │
│  依赖:                                                      │
│    ├── FlagEmbedding.FlagReranker  重排序模型               │
│    ├── BGE_RERANKER_LARGE          模型路径                 │
│    ├── BGE_RERANKER_DEVICE         运行设备                 │
│    └── BGE_RERANKER_FP16           半精度模式               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 节点设计要点

#### 要点 1：交叉编码器 vs 双塔模型的选择

```python
# 双塔模型（召回阶段）— 我们的向量检索用的就是这个
# Query 和 Document 独立编码，可预计算
query_vec = encoder(query)
doc_vec = encoder(doc)        # 可离线计算存储
score = cosine(query_vec, doc_vec)

# 交叉编码器（精排阶段）— Reranker 用的就是这个
# Query 和 Document 联合编码，精度更高
pairs = [(query, doc1), (query, doc2), ...]
scores = reranker.compute_score(pairs)  # 需要在线计算
```

召回阶段用双塔模型快速筛选候选集，精排阶段用交叉编码器深度判断相关性，两者配合形成"粗筛 → 精排"的经典 pipeline。

#### 要点 2：防御性取值

```python
# 所有字段用 get(key, "") 给默认值，再 .strip() 清洗
content = doc.get("content", "").strip()
title = doc.get("title", "").strip()
chunk_id = doc.get("chunk_id", "").strip() or None  # 空字符串转 None
```

上游数据可能缺字段、值为 None、或含空格，统一用 `get(key, "").strip()` 兜底。

#### 要点 3：断崖检测双阈值互补

```
高分区: 8.0 → 7.2  abs_gap=0.8✓  rel_gap=0.10✗  → 靠 abs_gap 截断
低分区: 0.8 → 0.5  abs_gap=0.3✗  rel_gap=0.375✓ → 靠 rel_gap 截断
负分区: -1.5→-1.8  abs_gap=0.3✗  rel_gap=0.2    → 正常工作（因为分母加了 abs）
```

`abs_gap` 和 `rel_gap` 用 `or` 关系互相补位，确保任何分数区间的断崖都能被检测到。

#### 要点 4：降级处理策略

```python
try:
    scores = reranker.compute_score(pairs)
    # 正常处理...
except Exception as e:
    # 降级：返回原序，score 设为 None
    return [{**item, "score": None} for item in doc_items]
```

Reranker 失败时系统不中断，降级为 RRF 的原始排序。下游断崖检测遇到 `score=None` 会 `continue` 跳过，不会报错，最终取 `max_topk` 条返回。
