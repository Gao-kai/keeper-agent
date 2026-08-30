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
import json
import logging
import re
from typing import List, Dict, Any, Set
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_query_config, QueryConfig
from knowledge.processor.query_process.exception import ValidationError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.log_config import setup_logging
from knowledge.utils.milvus_client import create_hybrid_search_request, execute_hybrid_search, get_milvus_client

logger = logging.getLogger("工具函数")


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


MAX_ENTITY_NAME_LENGTH: int = 15
ALIGN_QUERY_ENTITY_NAME_SCORE: float = 0.66
ALLOWED_ENTITY_LABELS_CN: set = {
	"设备(Device)"
	"部件(Part)"
	"操作(Operation)"
	"步骤(Step)"
	"警告(Warning)"
	"条件(Condition)"
	"工具(Tool)"
}

"""
EntityExtractor
基于LLM抽取用户问题中可能包含的实体名称，并完成清洗
"""


class EntityExtractor:
	
	def __init__(self):
		self._logger = logging.getLogger(self.__class__.__name__)
	
	def extract(self, rewritten_query: str) -> List[str]:
		if not rewritten_query:
			self._logger.warning(f"用户输入{rewritten_query}为空，无法进行LLM实体提取")
			return []
		
		llm_client = get_llm_client(response_json=True)
		if llm_client is None:
			self._logger.warning(f"LLM客户端连接失败 无法进行LLM实体提取")
			return []
		
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


class EntityAligner:
	def __init__(self):
		self._logger = logging.getLogger(self.__class__.__name__)
	
	def align(self, entities: List[str], item_names: List[str], config: QueryConfig) -> Dict[str, Any]:
		
		fallback_result = {
			"entity_names": [],  # 将LLM返回的实体名称经过Milvus查询后对齐的实体名称
			"entity_elements": []  # 将LLM返回的实体名称经过Milvus查询后对齐的实体信息
		}
		
		if not entities:
			return fallback_result
		
		# 1. LLM从用户问题中提取出来的所有实体进行向量化 获取混合向量（必须保证和入库时的嵌入模型一致性）
		bge_m3_embedding_model = get_bge_m3_embedding_model()
		if bge_m3_embedding_model is None:
			self._logger.error(f"嵌入模型BGE-M3不存在")
			return fallback_result
		
		milvus_client = get_milvus_client()
		if milvus_client is None:
			self._logger.error(f"Milvus客户端不存在")
			return fallback_result
		
		hybrid_embeddings: Dict[str, Any] = generate_hybrid_embeddings(
			embedding_model=bge_m3_embedding_model,
			embedding_docs=entities
		)
		
		if not hybrid_embeddings:
			self._logger.error(f"LLM模型提取的实体名称获取混合向量失败")
			return fallback_result
		
		self._logger.info(f"LLM提取出来的用户问题中的实体名称{entities} -> 成功生成BGE-M3向量嵌入模型混合向量 ")
		
		# 2. 拿到混合向量去milvus中查询 找到TOP K个最相似实体名称（限制在同一个item_names中）
		seen: set = set()
		aligned_entity_names = []
		aligned_entity_elements = []
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
					aligned_entity_elements.append(aligned_result)
		
		self._logger.info(f"用户问题中的实体对齐个数为:{len(aligned_entity_names)}")
		self._logger.info(f"用户问题中的实体对齐名字为:{aligned_entity_names}")
		
		return {
			"aligned_entity_names": aligned_entity_names,
			"aligned_entity_elements": aligned_entity_elements
		}
	
	def align_by_one(self, dense_vector, sparse_vector, milvus_client, config, entity_name, item_names) -> List[
		Dict[str, Any]]:
		expr = "item_name IN {item_names}"
		expr_params = {"item_names": item_names}
		
		# 构建混合查询
		reqs = create_hybrid_search_request(
			dense_vector=dense_vector,
			sparse_vector=sparse_vector,
			dense_req_field_name="entity_name_dense_vector",
			sparse_req_field_name="entity_name_sparse_vector",
			expr=expr,
			expr_params=expr_params,
			limit=5
		)
		
		# 执行混合查询
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
		
		if not hybrid_search_result or not hybrid_search_result[0]:
			self._logger.error(f"实体名称{entity_name}混合检索结果为空")
			return [
				{
					"original_entity_name": entity_name,
					"aligned_entity_name": "",
					"reason": f"实体名称{entity_name}混合检索结果为空"
				}
			]
		
		# 构建基于不同的商品名称->实体名称映射
		# 避免“请问万用表A和万用表B安装电池有啥不同？”大模型提取出的安装和电池两个实体名称匹配到不同商品名遗漏的问题
		best_entity_by_item_name = {}
		hits = hybrid_search_result[0]
		for hit in hits:
			entity = hit.get("entity")
			item_name = entity.get("item_name", "")
			# 相同商品名称的hit中天然得分高的排名靠前（因此天然保留同一个item_name下的最高得分hit）
			if item_name not in best_entity_by_item_name:
				best_entity_by_item_name[item_name] = hit
		
		# 遍历best_entity_by_item_name 构建返回结果
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
				}
			)
		
		if not results:
			return [
				{
					"original_entity_name": entity_name,
					"aligned_entity_name": "",
					"reason": f"实体名称{entity_name}混合检索结果得分过低相似性太低"
				}
			]
		
		return results


class QueryKnowledgeGraphNode(BaseNode):
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 1. 参数校验
		query_config = get_query_config()
		item_names, rewritten_query = self.validate_inputs(state)
		
		# 2. 知识图谱查询编排
		self.query_graph_pipeline(rewritten_query, item_names, query_config, state)
		
		return state
	
	def validate_inputs(self, state: QueryGraphState):
		item_names = state.get("item_names")
		rewritten_query = state.get("rewritten_query")
		
		if not item_names or not isinstance(item_names, list):
			raise ValidationError(node_name=self.name, message=f"输入参数 [item_names] 校验失败")
		
		if not rewritten_query or not isinstance(rewritten_query, str):
			raise ValidationError(node_name=self.name, message=f"输入参数 [rewritten_query] 校验失败")
		
		return item_names, rewritten_query
	
	def query_graph_pipeline(self, rewritten_query, item_names, query_config, state):
		"""
		
		Args:
			rewritten_query:
			item_names:
			query_config:
			state:

		Returns:

		"""
		self.log_step(step_name="STEP-2", message="抽取实体名称")
		# 1. 用户问题中抽取实体名称
		entity_extractor = EntityExtractor()
		cleaned_entities = entity_extractor.extract(rewritten_query)
		
		# 2. 基于抽取的实体名称和导入时存储在Milvus中的实体名称进行对齐
		entity_aligner = EntityAligner()
		entity_aligner.align(cleaned_entities, item_names, query_config)


if __name__ == "__main__":
	setup_logging()
	print("开始测试知识图谱查询节点")
	_state: QueryGraphState = {
		"rewritten_query": "H3C LA2608 室内无线网关怎么创建 WLAN-ESS 接口呢？",
		"item_names": ["H3C LA2608 室内无线网关"]
	}
	query_knowledge_node = QueryKnowledgeGraphNode()
	state = query_knowledge_node.process(_state)
	print(state)
