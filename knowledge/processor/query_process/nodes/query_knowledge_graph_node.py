"""
# 知识图谱查询节点

## 代码设计

导入侧（写）                              查询侧（读）
─────────────────────                    ─────────────────────
LLM 从文档中抽取实体             ←对应→     LLM 从问题中抽取实体
实体名向量化 → 写入 Milvus       ←对应→     实体名向量化 → 在 Milvus 中对齐
实体/关系 → 写入 Neo4j          ←对应→     在 Neo4j 中查种子节点、扩展关系
chunk_id 关联到 Entity         ←对应→     根据 chunk_id 从 Milvus 回填文本

## 问题
1. 为什么对齐的时候会返回得分相同的？
核心原因是在入库的时候只对单个chunk中的entity_name做了去重
多路并行处理的情况下一个实体名比如APP可能在CHUNK1和CHUNK2都加入到了向量库中
当我拿着用户的问题比如"APP怎么打开"，提取出来APP然后构建混合向量去检索的时候，自然就会检索到多个chunk

核心结论：distance 相同 ≠ 同一个实体。它既可能是"不同实体加权平均分恰好相等"（机制特性），也可能是"同一实体名被重复入库的多条记录"（数据问题）。后一种情况在你们的 find_best_entity 只看 hits[0] 的场景下影响不大，但如果后续要展示多条结果，建议：

入库侧跨 chunk 对 entity_name 做全局去重（如入库前按 item_name + entity_name 查重），避免重复记录；
查询侧按 entity_name 对 hybrid_search_result[0] 再做一次去重；
若需要更精确的融合，可考虑 norm_score=False 或调整两个检索器的权重，拉开分数区分度。

"""

# ================================================================
# 导入区
# ================================================================
import json
import logging
import re
from typing import List, Dict, Any, Set, Optional, NotRequired, TypedDict, cast

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_query_config, QueryConfig
from knowledge.processor.query_process.exception import ValidationError, Neo4jError, MilvusError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.log_config import setup_logging
from knowledge.utils.milvus_client import create_hybrid_search_request, execute_hybrid_search, get_milvus_client, \
	query_chunks_by_chunk_id_list
from knowledge.utils.neo4j_client import get_neo4j_client

# ================================================================
# 常量与模块级日志
# ================================================================
logger = logging.getLogger("工具函数")

# 种子节点权重
SEED_NODE_WEIGHT = 2

# 一跳范围内邻居节点权重
NBR_NODE_WEIGHT = 1

# 实体名称最大长度：LLM 抽取与入库截断均以此为上限
MAX_ENTITY_NAME_LENGTH: int = 15

# 实体对齐相似度阈值：低于该得分的弱匹配被过滤（Milvus distance）
ALIGN_QUERY_ENTITY_NAME_SCORE: float = 0.66

# LLM 抽取实体时允许的实体标签（注入系统提示词，约束输出类别）
# 注意：集合元素之间必须用逗号分隔，否则相邻字符串字面量会被 Python 隐式拼接成一个
ALLOWED_ENTITY_LABELS_CN: set = {
	"设备(Device)",
	"部件(Part)",
	"操作(Operation)",
	"步骤(Step)",
	"警告(Warning)",
	"条件(Condition)",
	"工具(Tool)",
}


# ================================================================
# 工具函数区：LLM 响应解析
# ================================================================
def parse_and_clean_llm_response(llm_response: str) -> List[str]:
	"""
	解析并清洗 LLM 返回的实体抽取结果。

	LLM 以 JSON 格式返回实体列表，但实际输出可能带有 markdown 代码围栏、
	非 JSON 噪音或格式异常，因此需要依次完成：
	1. 空值兜底：为空时直接返回空列表，避免下游解析报错；
	2. 格式清洗：剥离开头的 ```json/``` 代码围栏，还原纯 JSON 文本；
	3. 反序列化：将 JSON 文本解析为字典，失败则记录日志并返回空列表；
	4. 结构校验：提取 "entities" 键对应的列表，非列表则视为非法输出；
	5. 逐条清洗：过滤非字符串、超长截断、去除重复实体名。

	Args:
		llm_response: LLM 输出的原始字符串，期望为 JSON 格式（如 {"entities": [...]}）

	Returns:
		List[str]: 清洗去重后的实体名称列表；任何一步异常均返回空列表
	"""
	# 1. 空值兜底：LLM 未返回任何内容时直接返回空列表
	if not llm_response:
		return []
	
	# 2. 清洗 markdown 代码围栏：LLM 常把 JSON 包在 ```json ... ``` 中，需剥离开头结尾标记
	cleaned_text = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
	cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
	
	# 3. json反序列化可能失败：LLM 输出可能含多余文字，解析失败时降级为空列表
	try:
		data: Dict[str, Any] = json.loads(cleaned_text)
	except json.JSONDecodeError as e:
		logger.error(f"JSON反序列化失败: {e}")
		return []
	
	# 4. 获取实体列表：只认 "entities" 键，且必须是列表类型
	entity_names = data.get("entities", [])
	if not isinstance(entity_names, list):
		return []
	
	# 5. 截断和去重：逐条过滤非法项，超长截断，并用 set 保证实体名唯一
	cleaned_entities: List[str] = []
	seen: Set[str] = set()
	for entity_name in entity_names:
		# 为非字符串项：跳过，防止对非文本类型调用 len() 报错
		if not isinstance(entity_name, str):
			continue
		
		# 超出长度：按 MAX_ENTITY_NAME_LENGTH 截断，控制入库实体名长度
		if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
			entity_name = entity_name[:MAX_ENTITY_NAME_LENGTH]
		
		# 去重：仅保留首次出现的实体名，避免下游重复查询
		if entity_name and entity_name not in seen:
			seen.add(entity_name)
			cleaned_entities.append(entity_name)
	
	return cleaned_entities

# ================================================================
# 数据类型定义区：与 TS interface 对应的结构类型
# ================================================================
class Neo4jQueryPair(TypedDict):
	"""
	Neo4j 查询参数对：以 (item_name, entity_name) 唯一定位一个实体节点。
	与导入侧写入 Neo4j 时使用的键一致，保证读写对应。
	"""
	item_name: str
	entity_name: str


class OneHopRelation(TypedDict):
	"""
	一跳关系三元组：种子节点到邻居实体的一条有向关系。

	由 Neo4jGraphReader.query_one_hop_nodes 产出，对应图中
	"(tail_entity_name)-[:relation]->(tail_entity_name)" 这条关系；
	tail_entity_name/head_entity_name 的方向与图谱中箭头方向保持一致（通过 startNode(r) 判断）
	item_name 用于唯一标识所属商品。

	对应 TS 中形如的 interface：
	interface OneHopRelation {
		head_entity_name: string;       // 关系起点实体名（有向关系的箭头头部）
		tail_entity_name: string;       // 关系终点实体名
		item_name: string;  // 所属商品名
		relation: string;   // 关系类型（如 MENTIONED_IN 已被过滤）
	}
	"""
	head_entity_name: str
	tail_entity_name: str
	item_name: str
	relation: str


class AlignedEntityResult(TypedDict):
	"""
	单个实体名称的对齐结果（由 EntityAligner.align_by_one 产出，对应返回列表中的每个元素）。

	对应 TS 中形如的 interface：
	interface AlignedEntityResult {
		original_entity_name: string;   // LLM 抽取出的原始实体名
		aligned_entity_name: string;    // 对齐到知识库后的规范实体名，对齐失败时为空字符串
		reason: string;                 // 对齐过程/结果的说明（成功或失败原因）
		score: number | null;           // 相似度得分（Milvus 距离），失败时为 null
		source_chunk_id?: string;       // 命中的来源 chunk id（对齐成功时才有）
		context?: string;               // 命中实体的上下文文本（对齐成功时才有）
		item_name?: string;             // 命中所属商品名（对齐成功时才有）
	}
	"""
	original_entity_name: str
	aligned_entity_name: str
	reason: str
	score: Optional[float]
	source_chunk_id: NotRequired[str]
	context: NotRequired[str]
	item_name: NotRequired[str]


class EntityAlignResult(TypedDict):
	"""
	EntityAligner.align 的整体返回值。

	对应 TS 中形如的 interface：
	interface EntityAlignResult {
		aligned_entity_names: string[];          // 对齐成功的规范实体名列表
		aligned_entity_fields: AlignedEntityResult[];  // 每个实体的对齐详情
	}
	"""
	aligned_entity_names: List[str]
	aligned_entity_fields: List[AlignedEntityResult]


class MilvusChunkRow(TypedDict):
	"""
	从 Milvus chunks 集合按主键批量查询出的单行切片数据。

	由 ChunkBackFiller.search_chunk_node_in_milvus 产出（output_fields 决定返回哪些标量字段），
	即 query_graph_pipeline 返回结果中 chunks 列表的元素。

	对应 TS 中形如的 interface：
	interface MilvusChunkRow {
		chunk_id: number;    // Milvus 主键（INT64 auto_id 自增生成）
		item_name: string;   // 所属商品名
		content: string;     // 切片正文
		file_title: string;  // 来源文件名
		// 还可包含 Milvus 返回的其他标量字段（以调用方传入的 output_fields 为准）
	}
	"""
	chunk_id: int
	item_name: str
	content: str
	file_title: str


# Neo4j 种子节点
class Neo4jSeedRecord(TypedDict):
	item_name: str
	entity_name: str


class QueryGraphPipelineResult(TypedDict):
	"""
	query_graph_pipeline 的返回结果：整条图谱查询编排链路的最终产出，
	即 QueryKnowledgeGraphNode 图谱查询阶段交给下游（RRF 混合融合、答案生成 prompt、会话状态）的数据契约。

	对应 TS 中形如的 interface：
	interface QueryGraphPipelineResult {
		graph_chunks: MilvusChunkRow[];           // 按权重排序回填后的切片文本 → 送入 RRF 参与混合检索融合
		graph_relation_texts: string[];           // 一跳关系文本描述（形如 "A -[关系]-> B"）→ 送入答案生成 prompt
		seed_nodes: Neo4jSeedRecord[];      // Neo4j 精确/模糊命中的种子节点原始记录（列名 n.name / n.item_name）
		one_hop_nodes: OneHopRelation[];    // 种子节点一跳范围内的三元组（head/tail 方向与关系类型）
		llm_extract_entities: string[];     // LLM 从用户问题中抽取的实体名（清洗去重后）
		aligned_entity_names: string[];     // 对齐成功的规范实体名
		aligned_entity_fields: AlignedEntityResult[];  // 每个实体的对齐详情（得分/来源 chunk/上下文等）
	}
	"""
	graph_chunks: List[MilvusChunkRow]
	graph_relation_texts: List[str]
	seed_nodes: List[Neo4jSeedRecord]
	one_hop_nodes: List[OneHopRelation]
	llm_extract_entities: List[str]
	aligned_entity_names: List[str]
	aligned_entity_fields: List[AlignedEntityResult]


def transform_one_hop_nodes_to_text(one_hop_nodes: List[OneHopRelation]) -> List[str]:
	"""
	将一跳关系三元组列表转换为便于 LLM 理解的自然语言文本描述。

	每条三元组格式化为 "item_name head -[relation]-> tail"（无 item_name 时省略前缀），
	跳过字段不全的脏数据。

	Args:
		one_hop_nodes: 一跳关系三元组列表（元素为 OneHopRelation，
			含 head_entity_name/tail_entity_name/item_name/relation）

	Returns:
		List[str]: 关系文本描述列表；入参为空或全部为脏数据时返回空列表
	"""
	if not one_hop_nodes:
		return []
	
	relation_texts: List[str] = []
	for one_hop_node in one_hop_nodes:
		head_entity_name = one_hop_node.get("head_entity_name")
		tail_entity_name = one_hop_node.get("tail_entity_name")
		item_name = one_hop_node.get("item_name")
		relation = one_hop_node.get("relation")
		
		# 脏数据过滤：head/tail/relation 任一缺失则跳过该条三元组
		if not (head_entity_name and tail_entity_name and relation):
			continue
		
		if not item_name:
			relation_texts.append(f"{head_entity_name} -({relation})-> {tail_entity_name}")
		else:
			relation_texts.append(f"{item_name} {head_entity_name} -({relation})-> {tail_entity_name}")
	
	return relation_texts


# ================================================================
# 实体抽取器：LLM 从用户问题中抽取实体名称
# ================================================================
class EntityExtractor:
	"""
	实体抽取器：基于 LLM 从用户问题中抽取实体名称。

	工作流程：
	1. 空值兜底：用户问题为空或 LLM 客户端不可用时直接返回空列表；
	2. 构造提示词：将允许的实体标签（ALLOWED_ENTITY_LABELS_CN）与实体名
	   长度上限（MAX_ENTITY_NAME_LENGTH）注入系统提示词，约束 LLM 输出；
	3. 调用 LLM：要求以 JSON 格式返回 {"entities": [...]}；
	4. 清洗结果：通过 parse_and_clean_llm_response 完成代码围栏剥离、
	   反序列化、非字符串过滤、长度截断与去重。

	典型调用场景：QueryKnowledgeGraphNode.query_graph_pipeline 中先由此类
	从改写后的问题中抽取实体，再交给 EntityAligner 到 Milvus 实体名称
	集合中对齐。

	Attributes:
		_logger: 以类名命名的日志记录器
	"""
	
	def __init__(self):
		self._logger = logging.getLogger(self.__class__.__name__)
	
	def extract(self, rewritten_query: str) -> List[str]:
		"""
		从用户问题中抽取实体名称。

		1. 空问题或 LLM 客户端不可用 → 返回空列表；
		2. 以 ENTITY_EXTRACT_SYSTEM_PROMPT 为系统提示词，约束 LLM 以
		   JSON 格式返回 {"entities": [...]}；
		3. 返回结果经 parse_and_clean_llm_response 清洗去重。

		Args:
			rewritten_query: 大模型改写后的用户问题

		Returns:
			List[str]: 清洗后的实体名称列表；任何异常均返回空列表
		"""
		# 1. 空值兜底：用户问题为空无法抽取
		if not rewritten_query:
			self._logger.warning(f"用户输入{rewritten_query}为空，无法进行LLM实体提取")
			return []
		
		# 2. 客户端检查：LLM 客户端不可用则降级为空结果
		llm_client = get_llm_client(response_json=True)
		if llm_client is None:
			self._logger.warning(f"LLM客户端连接失败 无法进行LLM实体提取")
			return []
		
		# 3. 构造提示词并调用 LLM，返回后清洗
		try:
			human_message = f"用户问题：{rewritten_query}"
			system_message = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
				ALLOWED_ENTITY_LABELS_CN=ALLOWED_ENTITY_LABELS_CN,
				MAX_ENTITY_NAME_LENGTH=MAX_ENTITY_NAME_LENGTH
			)
			response = llm_client.invoke(
				[
					SystemMessage(content=system_message),
					HumanMessage(content=human_message)
				]
			)
			
			if not response:
				return []
			
			content = getattr(response, "content")
			cleaned_entities = parse_and_clean_llm_response(content)
			return cleaned_entities
		
		except Exception as e:
			self._logger.error(f"LLM提取实体名称报错:{e}", exc_info=True)
			return []


# ================================================================
# 实体对齐器：实体名称在 Milvus 中的检索对齐
# ================================================================
class EntityAligner:
	"""
	实体对齐器：将 LLM 抽取出的实体名称与知识库中的规范实体名称对齐。

	背景：
	导入侧将文档中的实体名称（含 BGE-M3 混合向量）写入 Milvus 实体名称集合；
	查询侧拿到用户问题中抽取的实体名称后，同样生成混合向量，在同一个集合中
	检索最相似的规范实体名，从而把用户口语化的实体说法映射到知识库的标准
	实体名（例如"万用表A/万用表B 装电池"中的"安装""电池"）。

	工作流程：
	1. 将待对齐的实体名称批量生成 BGE-M3 混合向量（须与入库侧模型保持一致）；
	2. 逐个实体执行混合检索，通过 item_name 过滤限定商品范围，取 TOP-K 候选；
	3. 按 item_name 分组，每组只保留得分最高的 hit，避免不同商品下的同名词
	   互相干扰；
	4. 过滤相似度低于 ALIGN_QUERY_ENTITY_NAME_SCORE 的弱匹配，并按
	   (entity_name, item_name) 去重。

	Attributes:
		_logger: 以类名命名的日志记录器
	"""
	
	def __init__(self):
		self._logger = logging.getLogger(self.__class__.__name__)
	
	def align(self, entities: List[str], item_names: List[str], config: QueryConfig) -> EntityAlignResult:
		"""
		将 LLM 抽取的实体名称与 Milvus 实体名称集合中的规范实体名对齐。

		1. 为空 / 模型缺失 / 客户端缺失 / 向量化失败 → 返回空结果的 fallback_result；
		2. 逐实体调用 align_by_one 做混合检索（限定 item_names 范围）；
		3. 按 (aligned_entity_name, item_name) 去重后汇总。

		Args:
			entities: LLM 抽取的实体名称列表
			item_names: 商品名称列表，用于限定检索范围
			config: 查询配置（含实体名称集合名等）

		Returns:
			EntityAlignResult: 对齐成功的实体名列表 + 每个实体的对齐详情列表
		"""
		# fallback：对齐失败/无输入时的空结果，键名与正常返回保持一致
		fallback_result: EntityAlignResult = {
			"aligned_entity_names": [],  # 将LLM返回的实体名称经过Milvus查询后对齐的实体名称
			"aligned_entity_fields": []  # 将LLM返回的实体名称经过Milvus查询后对齐的实体信息
		}
		
		# 1. 前置校验：实体列表为空直接返回
		if not entities:
			return fallback_result
		
		# 2. 依赖检查：嵌入模型与 Milvus 客户端必须可用
		bge_m3_embedding_model = get_bge_m3_embedding_model()
		if bge_m3_embedding_model is None:
			self._logger.error(f"嵌入模型BGE-M3不存在")
			return fallback_result
		
		milvus_client = get_milvus_client()
		if milvus_client is None:
			self._logger.error(f"Milvus客户端不存在")
			return fallback_result
		
		# 3. 实体向量化：必须保证和入库时的嵌入模型一致性
		hybrid_embeddings: Dict[str, Any] = generate_hybrid_embeddings(
			embedding_model=bge_m3_embedding_model,
			embedding_docs=entities
		)
		
		if not hybrid_embeddings:
			self._logger.error(f"LLM模型提取的实体名称获取混合向量失败")
			return fallback_result
		
		self._logger.info(f"LLM提取出来的用户问题中的实体名称{entities} -> 成功生成BGE-M3向量嵌入模型混合向量 ")
		
		# 4. 逐个实体检索对齐，并按 (aligned_entity_name, item_name) 去重汇总
		seen: set = set()
		aligned_entity_names = []
		aligned_entity_fields = []
		for index, entity_name in enumerate(entities):
			dense_vector = hybrid_embeddings["dense"][index]
			sparse_vector = hybrid_embeddings["sparse"][index]
			
			aligned_results = self.align_by_one(
				dense_vector=dense_vector,
				sparse_vector=sparse_vector,
				milvus_client=milvus_client,
				config=config,
				entity_name=entity_name,
				item_names=item_names
			)
			
			for aligned_result in aligned_results:
				
				aligned_entity_name = aligned_result.get("aligned_entity_name", "")
				item_name = aligned_result.get("item_name", "")
				unique_key = (aligned_entity_name, item_name)
				if unique_key not in seen:
					seen.add(unique_key)
					aligned_entity_names.append(aligned_entity_name)
					aligned_entity_fields.append(aligned_result)
		
		self._logger.info(f"用户问题中的实体对齐个数为:{len(aligned_entity_names)}")
		self._logger.info(f"用户问题中的实体对齐名字为:{aligned_entity_names}")
		
		return {
			"aligned_entity_names": aligned_entity_names,
			"aligned_entity_fields": aligned_entity_fields
		}
	
	def align_by_one(self, dense_vector, sparse_vector, milvus_client, config, entity_name, item_names) -> List[
		AlignedEntityResult]:
		"""
		单个实体名称的混合检索与对齐。

		1. 构造混合检索请求（稠密 + 稀疏向量，权重 0.5/0.5，norm_score=True）；
		2. 在 entity_name_collection 中按 "item_name IN ..." 过滤检索 TOP-K；
		3. 同一商品名称下只保留得分最高的 hit（避免同实体名重复入库记录干扰）；
		4. 得分低于 ALIGN_QUERY_ENTITY_NAME_SCORE 的弱匹配被过滤。

		Returns:
			List[AlignedEntityResult]: 对齐结果；检索为空或得分过低时返回带 reason 的失败条目
		"""
		# 1. 构造查询表达式：限定商品范围
		expr = "item_name IN {item_names}"
		expr_params = {"item_names": item_names}
		
		# 2. 构建混合查询请求
		reqs = create_hybrid_search_request(
			dense_vector=dense_vector,
			sparse_vector=sparse_vector,
			dense_req_field_name="entity_name_dense_vector",
			sparse_req_field_name="entity_name_sparse_vector",
			expr=expr,
			expr_params=expr_params,
			limit=5
		)
		
		# 3. 执行混合查询
		hybrid_search_result = execute_hybrid_search(
			milvus_client=milvus_client,
			limit=5,
			reqs=reqs,
			collection_name=config.entity_name_collection,
			output_fields=[
				"entity_name",
				"source_chunk_id",
				"item_name",
				"context"
			],
			ranker_weights=[0.5, 0.5],
			norm_score=True
		)
		self._logger.info(f"实体名称{entity_name}检索完成")
		
		# 4. 检索结果为空时返回失败条目
		if not hybrid_search_result or not hybrid_search_result[0]:
			self._logger.error(f"实体名称{entity_name}混合检索结果为空")
			return [
				{
					"original_entity_name": entity_name,
					"aligned_entity_name": "",
					"score": None,
					"reason": f"实体名称{entity_name}混合检索结果为空"
				}
			]
		
		"""
		EDGE CASE
		用户问题: “请问万用表A和万用表B安装电池有啥不同？”
		确定商品名：["万用表A"，"万用表B"]
		LLM基于用户问题抽取：["安装","电池"]
		
		对“安装”实体进行混合检索，由于标量字段检索过滤条件是"item_name IN {item_names}"
		所以对"万用表A"，"万用表B"商品下的实体都会进行检索
		假设检索出来hits:
		1. 万用表A中的chunk1中“安装”得分是0.88
		2. 万用表A中的chunk2中“安装”得分是0.85
		3. 万用表B中的chunk1中“安装”得分是0.83
		
		此时假设机械的只取hits中得分最高的1，将会把万用表B中的chunk1遗漏
		所以这里先以商品名称为key，以该商品最高得分hit为value构建map，避免遗漏
		"""
		
		# 5. 构建基于不同的商品名称 -> 实体名称映射
		best_entity_by_item_name = {}
		hits = hybrid_search_result[0]
		for hit in hits:
			entity = hit.get("entity")
			item_name = entity.get("item_name", "")
			# 相同商品名称的hit中天然得分高的排名靠前（因此天然保留同一个item_name下的最高得分hit）
			if item_name not in best_entity_by_item_name:
				best_entity_by_item_name[item_name] = hit
		
		# 6. 遍历best_entity_by_item_name 构建返回结果，过滤得分过低的弱匹配
		results = []
		for item_name, best_hit_by_item in best_entity_by_item_name.items():
			distance = best_hit_by_item.get("distance")
			
			if float(distance) <= ALIGN_QUERY_ENTITY_NAME_SCORE:
				continue
			
			best_item_entity_fields = best_hit_by_item.get("entity", {})
			results.append(
				{
					"original_entity_name": entity_name,
					"aligned_entity_name": best_item_entity_fields.get("entity_name", ""),
					"reason": f"商品 {best_item_entity_fields.get("item_name")} 中最相似的实体名称",
					"source_chunk_id": best_item_entity_fields.get("source_chunk_id", ""),
					"context": best_item_entity_fields.get("context", ""),
					"item_name": best_item_entity_fields.get("item_name", ""),
					"score": distance
				}
			)
		
		# 7. 全部未过阈值时返回失败条目
		if not results:
			return [
				{
					"original_entity_name": entity_name,
					"aligned_entity_name": "",
					"score": None,
					"reason": f"实体名称{entity_name}混合检索结果得分过低相似性太低"
				}
			]
		
		return results


# ================================================================
# 查询编排节点：实体抽取 → 实体对齐 → Neo4j 种子节点查询
# ================================================================



class QueryKnowledgeGraphNode(BaseNode):
	"""
	知识图谱查询节点（查询工作流中的一个图节点）。

	职责：在查询阶段读取知识图谱，主要完成两件事——
	1. 实体抽取：通过 EntityExtractor 用 LLM 从改写后的问题中抽取实体名称；
	2. 实体对齐：通过 EntityAligner 将抽取出的实体在 Milvus 实体名称集合中
	   检索对齐，映射到知识库中的规范实体名，供下游图查询使用。

	与导入侧（knowledge_graph_node）形成读写对应：
	  导入（写）: 实体名向量化 → 写 Milvus；实体/关系 → 写 Neo4j
	  查询（读）: 实体名向量化 → 在 Milvus 对齐 → 在 Neo4j 扩展关系 → 回填文本

	依赖 state:
		item_names: 商品名称列表，用于限定检索范围
		rewritten_query: 大模型改写后的用户问题

	输出 state:
		aligned_entities: 实体对齐结果列表（AlignedEntityResult）
		kg_seed_nodes: 在 Neo4j 中定位到的种子节点列表（{"item_name", "entity_name"}）
	"""
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 1. 参数校验
		query_config = get_query_config()
		item_names, rewritten_query = self.validate_inputs(state)
		
		# 2. 知识图谱查询编排
		pipeline_result = self.query_graph_pipeline(rewritten_query, item_names, query_config, state)
		
		return state
	
	def validate_inputs(self, state: QueryGraphState):
		"""
		校验查询节点必需的输入参数。

		Args:
			state: 查询图共享状态

		Returns:
			tuple: (item_names, rewritten_query)

		Raises:
			ValidationError: item_names 或 rewritten_query 缺失/类型不正确
		"""
		item_names = state.get("item_names")
		rewritten_query = state.get("rewritten_query")
		
		if not item_names or not isinstance(item_names, list):
			raise ValidationError(node_name=self.name, message=f"输入参数 [item_names] 校验失败")
		
		if not rewritten_query or not isinstance(rewritten_query, str):
			raise ValidationError(node_name=self.name, message=f"输入参数 [rewritten_query] 校验失败")
		
		return item_names, rewritten_query
	
	def query_graph_pipeline(self, rewritten_query, item_names, query_config,
                         state) -> QueryGraphPipelineResult:
		"""
		知识图谱查询编排主流程。

		步骤：
		1. 实体抽取：通过 EntityExtractor + LLM 从改写后的问题中抽取实体名称；
		2. 实体对齐：通过 EntityAligner 将抽取的实体与 Milvus 中的规范实体
		   名称对齐；
		3. 构造 Neo4j 查询参数对：(item_name, entity_name) 与入库键保持一致；
		4. 查询种子节点：通过 Neo4jGraphReader 精确/模糊查询，格式化结果。

		Args:
			rewritten_query: 大模型改写后的用户问题
			item_names: 商品名称列表，用于限定实体检索范围
			query_config: 查询配置（含实体名称集合名等）
			state: 查询图共享状态（本方法会原地写入 aligned_entities 与 kg_seed_nodes）

		Returns:
			None：结果通过修改 state 透出
		"""
		self.log_step(step_name="STEP-2", message="开始基于用户问题从LLM中提取抽取实体名称")
		# 1. 用户问题中抽取实体名称
		entity_extractor = EntityExtractor()
		cleaned_entities = entity_extractor.extract(rewritten_query)
		self.log_step(step_name="STEP-2", message=f"基于用户问题从LLM中提取抽取实体名称:{cleaned_entities}")
		
		# 2. 基于抽取的实体名称和导入时存储在Milvus中的实体名称进行对齐
		entity_aligner = EntityAligner()
		self.log_step(step_name="STEP-3", message="基于LLM抽取实体名称在Milvus数据库中进行对齐")
		entity_aligned_result = entity_aligner.align(cleaned_entities, item_names, query_config)
		self.log_step(step_name="STEP-3", message=f"基于LLM抽取实体名称对齐结果为：{entity_aligned_result}")
		
		# 3. 基于对齐的实体名称及其信息进行格式化
		# 保证传递给Neo4j的是(item_name,entity_name)的参数对 因为存入Neo4j的时候key就是该参数对组成的 保证查询一体化
		aligned_entity_names = entity_aligned_result.get("aligned_entity_names", [])
		aligned_entity_fields = entity_aligned_result.get("aligned_entity_fields", [])
		query_pairs = self.get_neo4j_query_pairs(aligned_entity_fields)
		self.log_step(step_name="STEP-4", message=f"传递给Neo4j中查询的参数对为：{query_pairs}")
		
		# 4. 基于查询信息在Neo4j知识图谱中查询种子节点
		neo4j_graph_reader = Neo4jGraphReader(config=query_config)
		seed_nodes = neo4j_graph_reader.find_seed_nodes(query_pairs)
		cleaned_seed_nodes = []
		for seed_node in seed_nodes:
			item_name = seed_node.get("item_name")
			entity_name = seed_node.get("name")
			cleaned_seed_nodes.append({
				"item_name": item_name,
				"entity_name": entity_name
			})
		self.log_step(step_name="STEP-5", message=f"查询到的种子节点为：{cleaned_seed_nodes}")
		
		# 5. 查询每个种子节点一跳范围内的所有实体节点和关系
		one_hop_nodes = neo4j_graph_reader.query_one_hop_nodes(cleaned_seed_nodes)
		self.log_step(step_name="STEP-6", message=f"查询到的种子节点在其一跳范围内的节点和关系为：{one_hop_nodes}")
		
		# 6. 对所有种子节点和一跳范围内的所有节点进行加权 种子2 邻居节点1
		nodes_weight_list = neo4j_graph_reader.collect_nodes_by_weight(cleaned_seed_nodes, one_hop_nodes)
		self.log_step(step_name="STEP-7", message=f"对种子节点和一跳节点加权后的字典：{nodes_weight_list}")
		
		# 7. 基于加权后的节点和MENTION_IN关系查询出所有关联的chunk_id
		chunk_nodes_sorted = neo4j_graph_reader.query_chunk_id_by_node_weight(nodes_weight_list)
		self.log_step(step_name="STEP-8",
		              message=f"基于加权后的实体字典分组排序查询到所有MENTION_IN到的CHUNK结果是：{chunk_nodes_sorted}")
		
		# 8. 从Neo4j中查询分组后排序的chunk去Milvus向量数据库的chunks表查询对应的chunk
		chunk_back_filler = ChunkBackFiller(config=query_config)
		chunk_results = chunk_back_filler.search_chunk_node_in_milvus(chunk_nodes_sorted)
		self.log_step(step_name="STEP-9",
		              message=f"：从Neo4j中查询分组后排序的chunk去Milvus向量数据库反查到的chunk是: {chunk_results}")
		
		# 9. 将一跳范围内的所有三元组转化为文本结构拼接 方便大模型知道节点之间的关系
		relation_texts = transform_one_hop_nodes_to_text(one_hop_nodes)
		
		# 返回：通过 cast 断言字典字面量符合 QueryGraphPipelineResult 结构
		# （边界值 chunk_results/seed_nodes 源于 Milvus 行 dict 与 Neo4j Record，
		#  运行时形状与 TypedDict 定义一致，但静态类型上并非 TypedDict 实例，故显式断言）
		return {
			"graph_chunks": chunk_results,  # 回填后的切片文本 → 送入 RRF
			"graph_relation_texts": relation_texts,  # 关系文本描述 → 送入答案生成 prompt
			"seed_nodes": cleaned_seed_nodes,
			"one_hop_nodes": one_hop_nodes,
			"llm_extract_entities": cleaned_entities,
			"aligned_entity_names": aligned_entity_names,
			"aligned_entity_fields": aligned_entity_fields,
		}
	
	
	@staticmethod
	def get_neo4j_query_pairs(aligned_entity_fields: List[AlignedEntityResult]) -> List[Neo4jQueryPair]:
		"""
		将对齐结果转换为 Neo4j 查询参数对列表。

		规则：同一商品名称下实体名称不允许重复，不同商品名称下实体名称允许重复，
		最终以 (item_name, entity_name) 去重。

		Args:
			aligned_entity_fields: 实体对齐详情列表

		Returns:
			List[Neo4jQueryPair]: 去重后的查询参数对列表
		"""
		query_pairs: List[Neo4jQueryPair] = []
		seen: set = set()
		for aligned_entity_field in aligned_entity_fields:
			item_name = aligned_entity_field.get("item_name", "").strip()
			entity_name = aligned_entity_field.get('aligned_entity_name', "").strip()
			if not item_name or not entity_name:
				continue
			
			# 同一商品名称下实体名称不允许重复
			# 不同商品名称下实体名称允许重复
			unique_key = (item_name, entity_name)
			if unique_key not in seen:
				seen.add(unique_key)
				query_pairs.append({
					"item_name": item_name,
					"entity_name": entity_name
				})
		
		return query_pairs


# ================================================================
# Neo4j 图查询器：种子节点的精确/模糊查询
# ================================================================
class Neo4jGraphReader:
	"""
	Neo4j 图查询器：在知识图谱中定位种子节点。

	核心逻辑：
	1. 精确查询优先：以 (item_name, entity_name) 精确匹配 ENTITY 节点；
	2. 模糊查询兜底：精确未命中时，用 toLower + CONTAINS 做模糊匹配（最多 3 条）；
	3. 数量上限：返回的种子节点数不超过 max_seed_per_node。

	对齐规则（与 EntityAligner 保持一致）：
	- 同一商品名称下只保留一个实体（得分最高的）；
	- 不同商品名称下可以保留多个实体。
	例如：电池 -> [A, B, C]；安装 -> [B, C]，
	最终可得到 (A,电池)(B,电池)(B,安装)(C,电池)(C,安装) 等组合。

	Attributes:
		database: Neo4j 数据库名
		_max_seed_candidates: 种子节点数量上限
		_max_total_seeds: 总种子节点上限
		neo4j_driver: Neo4j 驱动实例
	"""
	
	def __init__(self, config: QueryConfig):
		self.logger = logging.getLogger(self.__class__.__name__)
		self.database = config.neo4j_database
		self._max_seed_candidates = config.kg_max_seed_candidates  # 每个实体最大种子节点候选数
		self._max_total_seeds = config.kg_max_total_seeds  # 总种子节点上限
		self._max_triples_per_seed = config.kg_max_triples_per_seed  # 每个种子最大三元组数
		self._max_total_triples = config.kg_max_total_triples  # 所有种子节点最大三元组数
		self._max_total_chunks = config.kg_max_total_chunks  # 总切片上限
		# 获取neo4j客户端及数据库
		self.neo4j_driver = get_neo4j_client()
		if self.neo4j_driver is None:
			raise Neo4jError(node_name="Neo4jGraphReader查询类", message="Neo4j客户端连接失败")
	
	def find_seed_nodes(self, query_pairs: List[Neo4jQueryPair]):
		"""
		查询种子节点：精确匹配优先，失败后模糊查询兜底。

		Args:
			query_pairs: (item_name, entity_name) 查询参数对列表

		Returns:
			list[Record]: Neo4j 查询返回的记录列表（列名为 n.name / n.item_name），
			数量上限为 max_seed_per_node；入参为空或查询异常时返回空列表
		"""
		if not query_pairs:
			self.logger.error("用于查询Neo4j图数据库的数据不存在")
			return []
		
		# 遍历query_pairs开始依次查询种子节点
		seed_nodes = []
		try:
			for query_pair in query_pairs:
				item_name = query_pair.get("item_name", "")
				entity_name = query_pair.get("entity_name", "")
				
				# 1. 精确查询操作（读取种子节点；实体节点的属性名与导入侧保持一致为 name）
				accurate_query_result = self.neo4j_driver.execute_query(
					query_="""
						MATCH (n:ENTITY {item_name:$item_name, name:$entity_name})
						RETURN n.name, n.item_name
						""",
					parameters_={
						"entity_name": entity_name,
						"item_name": item_name
					},
					database_=self.database
				)
				
				# 精确命中则收集记录并跳过模糊查询
				if accurate_query_result.records:
					seed_nodes.extend(accurate_query_result.records)
					continue
				
				# 2. 精确查询失败 → 模糊查询3条兜底
				fuzzy_query_result = self.neo4j_driver.execute_query(
					query_="""
						MATCH (n:ENTITY)
						WHERE n.item_name = $item_name AND toLower(n.name) CONTAINS toLower($entity_name)
						RETURN n.name, n.item_name
						LIMIT $limit
						""",
					parameters_={
						"entity_name": entity_name,
						"item_name": item_name,
						"limit": self._max_seed_candidates
					},
					database_=self.database
				)
				
				# 模糊查询同样取 records（EagerResult 不是可迭代对象，不能直接 extend）
				seed_nodes.extend(fuzzy_query_result.records)
		except Exception as e:
			self.logger.error(f"查询种子节点报错:{e}")
		
		return seed_nodes[:self._max_total_seeds]
	
	def query_one_hop_nodes(self, cleaned_seed_nodes: List[Neo4jQueryPair]) -> List[OneHopRelation]:
		"""
		从某个种子节点出发，查询其一跳范围内的所有邻居实体节点和关系。

		1. 过滤 MENTIONED_IN 的关系节点（仅保留实体间关系）；
		2. 双向查询：以种子结束和以种子开始都进行查询；
		3. 记录 head 和 tail 方向，保证与图谱中箭头一致（startNode(r) 判断）；
		4. 跨种子节点按 (item_name, head, tail, relation) 去重。

		例如：
		A种子节点 无向查询 A-B
		B种子节点 无向查询 B-A

		Args:
			cleaned_seed_nodes: 对齐后的种子节点列表（(item_name, entity_name)）

		Returns:
			List[OneHopRelation]: 一跳关系三元组列表，数量上限为 _max_total_triples；
			入参为空或查询异常时返回空列表
		"""
		one_hop_results: List[OneHopRelation] = []
		if not cleaned_seed_nodes:
			self.logger.error("种子节点数据不存在，无法查询其一跳节点")
			return []
		
		try:
			seen: set = set()
			for seed_node in cleaned_seed_nodes:
				entity_name = seed_node.get("entity_name", "")
				item_name = seed_node.get("item_name")
				seed_node_one_hoop_results = self.neo4j_driver.execute_query(
					query_="""
					MATCH (seed:ENTITY {item_name:$item_name,name:$entity_name})-[r]-(nbr:ENTITY)
					WHERE type(r) <> "MENTIONED_IN" AND nbr.item_name = $item_name
					RETURN
						CASE
							WHEN startNode(r) = seed THEN seed.name
							ELSE nbr.name
						END AS head,
						type(r) AS relation,
						CASE
							WHEN startNode(r) = seed THEN nbr.name
							ELSE seed.name
						END AS tail
					LIMIT $limit
					""",
					database_=self.database,
					parameters_={
						"entity_name": entity_name,
						"item_name": item_name,
						"limit": self._max_triples_per_seed
					}
				)
				
				# 遍历查询结果，统一转为 OneHopRelation 字典结构（不直接混入原始 Record）
				for one_hop_record in seed_node_one_hoop_results.records:
					head_entity_name = one_hop_record.get("head", "")
					tail_entity_name = one_hop_record.get("tail", "")
					relation = one_hop_record.get("relation", "")
					
					unique_key: tuple[str, str, str, str] = (
						item_name,
						head_entity_name,
						tail_entity_name,
						relation
					)
					
					if unique_key not in seen:
						seen.add(unique_key)
						one_hop_results.append({
							"head_entity_name": head_entity_name,
							"tail_entity_name": tail_entity_name,
							"item_name": item_name,
							"relation": relation
						})
			
			return one_hop_results[:self._max_total_triples]
		except Exception as e:
			self.logger.error(f"查询种子节点的一跳范围内关系节点失败{e}")
		return one_hop_results
	
	def collect_nodes_by_weight(self, cleaned_seed_nodes: List[Neo4jQueryPair], one_hop_nodes: List[OneHopRelation]) -> \
			List[Dict[str, Any]]:
		"""
		权重固定：只看节点身份 是种子节点还是邻居节点（本项目使用）
		权重累加：如果一个节点即是种子节点、又是邻居节点、或者在多条关系中反复出现，那么权重可以累加
		🙋两者对比的实现方法、利弊和适用场景
		Args:
			cleaned_seed_nodes:
			one_hop_nodes:

		Returns:

		"""
		
		# 处理种子节点
		if not cleaned_seed_nodes:
			self.logger.warning("种子节点为空")
		
		nodes_weight_map = {}
		for seed_node in cleaned_seed_nodes:
			item_name = seed_node.get("item_name", "")
			entity_name = seed_node.get("entity_name", "")
			key = (item_name, entity_name)
			if key not in nodes_weight_map:
				nodes_weight_map[key] = SEED_NODE_WEIGHT
		
		if not one_hop_nodes:
			self.logger.warning("一跳范围内节点为空")
		
		# 处理一跳范围内的邻居节点
		for one_hop_node in one_hop_nodes:
			item_name = one_hop_node.get("item_name", "")
			head_entity_name = one_hop_node.get("head_entity_name", "")
			tail_entity_name = one_hop_node.get("tail_entity_name", "")
			
			if head_entity_name and (item_name, head_entity_name) not in nodes_weight_map:
				nodes_weight_map[(item_name, head_entity_name)] = NBR_NODE_WEIGHT
			
			if tail_entity_name and (item_name, tail_entity_name) not in nodes_weight_map:
				nodes_weight_map[(item_name, tail_entity_name)] = NBR_NODE_WEIGHT
		
		# 返回格式化后的字典列表
		return [
			{
				"item_name": item_name,
				"entity_name": entity_name,
				"weight": weight
			}
			for (item_name, entity_name), weight
			in nodes_weight_map.items()
		]
	
	def query_chunk_id_by_node_weight(self, nodes_weight_list):
		"""
		基于加权后的节点信息反查节点对应的 chunk_id。
		最后返回的 chunk_id 一定是按"权重得分最高、被最多实体节点提及"的顺序排列的
		（排序规则见下方 Cypher 解释）。

		为什么必须做这一步反查，而不能直接拿实体节点上的 chunk_id 去 Milvus 做标量过滤？
		1. 实体节点上的 source_chunk_id 只在节点首次创建时写入（导入侧 ON CREATE 分支），
		   同一实体在其它 chunk 再次出现时（ON MATCH 分支）不会更新它，
		   因此它只代表实体"首次出现"的 chunk，会漏掉该实体出现的其它所有 chunk；
		2. 而 ENTITY -[:MENTIONED_IN]- CHUNK 关系在实体每次出现的 chunk 上都会建立一条，
		   才是"实体 ↔ 全部出现 chunk"的完整映射，反查正是沿这条关系补齐所有相关 chunk；
		3. 打分与排序（权重求和 + 提及次数）可以下推给图数据库聚合完成，
		   而 Milvus 的标量过滤只能筛出命中的 chunk，拿不到这份加权排序；
		4. 附带：Milvus 主键 chunk_id 为 INT64 自增 id，链路中存储的是其字符串形式，
		   即便要直接过滤也需先做类型转换，不如走图谱反查一步到位。

		打分逻辑（图结构加权聚合）：
		1. 入参 nodes_weight_list 是 collect_nodes_by_weight 产出的"实体节点 + 权重"列表，
		   每个元素形如 {"item_name":..., "entity_name":..., "weight":...}；
		   权重已在上一阶段直接指定（种子节点 SEED_NODE_WEIGHT / 一跳邻居 NBR_NODE_WEIGHT），
		   并非本查询计算得出。
		2. 本查询通过 ENTITY -[:MENTIONED_IN]- CHUNK 关系，把每个实体节点映射到它被提及的文本块，
		   再按 chunk 分组，将"所有提及该 chunk 的实体节点的权重之和"作为该 chunk 的总分。
		   即：实体节点的权重越高，它提及的 chunk 得分越高（实体权重 → chunk 的加权聚合，
		   查询内部不存在实体到实体的多跳传播）。

		Cypher 语句逐句解释：
		- UNWIND $nodes_weight_list AS node_with_weight
			把权重列表摊开成多行，每行对应一个 (entity_name, item_name, weight)；
		- MATCH (entity:ENTITY {name:node_with_weight.entity_name, item_name:node_with_weight.item_name})
			精确匹配 ENTITY 节点，用 name + item_name 双条件定位
			（item_name 用于区分不同文档/商品下的同名实体）；
		- -[:MENTIONED_IN]-(chunk:CHUNK {item_name:node_with_weight.item_name})
			无向关系匹配：从实体沿 MENTIONED_IN 边找到它"被提及于"的 CHUNK 节点（限定同 item_name）；
			一个实体可能提及多个 chunk，因此行数会扇出；
		- WITH chunk, sum(node_with_weight.weight) AS total_score, count(entity) AS total_counts
			按 chunk 分组聚合：
			  total_score  = 提及该 chunk 的所有实体节点权重之和（实体权重越高，chunk 总分越高）；
			  total_counts = 提及该 chunk 的实体个数（反映覆盖度/提及次数）；
		- RETURN chunk.chunk_id AS chunk_id, chunk.item_name AS item_name, total_score, total_counts
			输出每个 chunk 的标识与得分；
		- ORDER BY total_score DESC, total_counts DESC, chunk_id ASC
			按总分降序；同分按提及次数降序；再同分按 chunk_id 升序（保证排序结果确定性）；
		- LIMIT $limit
			只取分数最高的前 $limit 个 chunk（即 self._max_total_chunks）。

		Args:
			nodes_weight_list: 带权重的实体节点列表，形如
				[{"item_name": str, "entity_name": str, "weight": float}, ...]，
				由 collect_nodes_by_weight 生成；为空时直接返回 []。

		Returns:
			chunk_id_list: 反查得到的 chunk_id 列表。
				注意：当前实现中下方 for 循环仅做局部赋值、未 append 进 chunk_id_list，
				故实际恒返回 []，属待修复的遗留问题。
		"""
		if not nodes_weight_list:
			self.logger.info("加权节点列表为空")
			return []
		
		chunk_nodes_sorted = []
		
		try:
			group_by_score_hits = self.neo4j_driver.execute_query(
				database_=self.database,
				query_="""
				UNWIND $nodes_weight_list AS node_with_weight
				MATCH (entity:ENTITY {name:node_with_weight.entity_name,item_name:node_with_weight.item_name})
					  -[:MENTIONED_IN]-(chunk:CHUNK {item_name:node_with_weight.item_name})
				WITH chunk,sum(node_with_weight.weight) AS total_score,count(entity) AS total_counts
				RETURN chunk.chunk_id AS chunk_id,chunk.item_name AS item_name,total_score,total_counts
				ORDER BY total_score DESC,total_counts DESC,chunk_id ASC
				LIMIT $limit
				""",
				parameters_={
					"limit": self._max_total_chunks,
					"nodes_weight_list": nodes_weight_list
				}
			)
			
			"""
			加权之后假设有3个ENTITY节点
			A 2 A总计有5个chunk MENTIONED_IN 比如是CHUNK 12345
			B 2 B总计有3个chunk MENTIONED_IN 比如是CHUNK 12
			C 1 C总计有2个chunk MENTIONED_IN 比如是CHUNK 35
			
			按照chunk进行分组WITH(ORDER BY)
			对于chunk 1来说 AB两个ENTITY命中 权重累计是4 次数是2
			对于chunk 2来说 AB两个ENTITY命中 权重累计是4 次数是2
			对于chunk 3来说 BC两个ENTITY命中 权重累计是3 次数是2
			对于chunk 4来说 A一个ENTITY命中 权重累计是2 次数是1
			对于chunk 5来说 AC两个ENTITY命中 权重累计是3 次数是2
			
			最后排序：
			得分最高的权重是4
			提及次数最多的是2
			最终排序：
			chunk1 返回chunk1的id chunk1的item_name商品名、该组总得分权重累计4、提及总次数2（数据会汇总在一个Record中）
			chunk2
			chunk5
			chunk3
			chunk4
			
			"""
			for record in group_by_score_hits.records:
				chunk_id = record.get("chunk_id", "")
				item_name = record.get("item_name", "")
				total_score = record.get("total_score", "")
				if chunk_id and item_name:
					"""
					向量检索（embedding_chunks）和 HyDE 检索（hyde_embedding_chunks）从 Milvus hybrid_search 返回的结构也是这样的。
					保持三路来源格式统一，RRF 节点不需要做特殊处理就能直接消费。
					id 设为 None 是因为 kg 这路不经过 Milvus 搜索没有主键。
					Milvus查询返回的id是数据库中的主键ID自动生成的，source_chunk_id才是真正的
					"""
					chunk_nodes_sorted.append({
						"id": "None",
						"distance": float(total_score or 0.0),
						"entity": {
							"chunk_id": str(chunk_id),
							"item_name": str(item_name)
						}
						
					})
		
		except Exception as e:
			self.logger.error(f"基于加权后的节点信息反查节点对应的chunk_id异常：{e}")
		
		return chunk_nodes_sorted


class ChunkBackFiller:
	def __init__(self, config: QueryConfig):
		self.logger = logging.getLogger(self.__class__.__name__)
		self.milvus_client = get_milvus_client()
		self.config = config
		if self.milvus_client is None:
			raise MilvusError(node_name="ChunkBackFiller查询类", message="Milvus客户端连接失败")
	
	def search_chunk_node_in_milvus(self, chunk_nodes_sorted: List[Dict[str, Any]]):
		"""
		根据上一步反查到的 chunk_id 列表
		从 Milvus CHUNKS_COLLECTION 批量回填切片文本内容
		
		Args:
			chunk_nodes_sorted:

		Returns:

		"""
		
		# 获取chunk id列表
		if not chunk_nodes_sorted:
			return []
		chunk_id_list = []
		for chunk in chunk_nodes_sorted:
			entity = chunk.get("entity", {})
			chunk_id = str(entity.get('chunk_id', ""))
			if not chunk_id:
				continue
			try:
				chunk_id_list.append(int(chunk_id))
			except (ValueError, TypeError):
				chunk_id_list.append(chunk_id)
		
		self.logger.info(f"chunk_id 列表:{chunk_id_list}")
		
		if not chunk_id_list:
			return []
		
		# 查询Milvus数据库
		chunk_rows = query_chunks_by_chunk_id_list(
			milvus_client=self.milvus_client,
			collection_name=self.config.chunks_collection,
			chunk_id_list=chunk_id_list,
			output_fields=[
				"chunk_id",
				"item_name",
				"content",
				"file_title",
			]
		)
		
		# 建立chunk_id to chunk_row的映射表
		chunk_id_to_row_map = {}
		for chunk_row in chunk_rows:
			chunk_id = chunk_row.get("chunk_id")
			chunk_id_to_row_map[str(chunk_id)] = chunk_row
		
		# 因为Milvus基于主键ids查询的结果和向量查询的结果不同 不包含distance的排序结果
		# 因此需要按照chunk_nodes_sorted的顺序进行二次排序
		# 排序后最靠前的chunk保证和Neo4j中查询出来的权重最高的顺序一致
		chunk_results = []
		for sorted_chunk_node in chunk_nodes_sorted:
			entity = sorted_chunk_node.get("entity", {})
			chunk_id = str(entity.get("chunk_id", ""))
			chunk = chunk_id_to_row_map.get(chunk_id)
			
			if chunk is None:
				continue
			chunk_results.append(chunk)
		
		return chunk_results


# ================================================================
# 本地测试
# ================================================================
if __name__ == "__main__":
	setup_logging()
	print("开始测试知识图谱查询节点")
	_state: QueryGraphState = {
		"rewritten_query": "H3C LA2608 室内无线网关怎么创建 WLAN-ESS 接口呢？",
		"item_names": ["H3C LA2608 室内无线网关"]
	}
	query_knowledge_node = QueryKnowledgeGraphNode()
	_state = query_knowledge_node.process(_state)
	print(_state)
