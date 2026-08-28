"""
# 知识图谱查询节点

## 代码设计

导入侧（写）                              查询侧（读）
─────────────────────                    ─────────────────────
LLM 从文档中抽取实体             ←对应→     LLM 从问题中抽取实体
实体名向量化 → 写入 Milvus       ←对应→     实体名向量化 → 在 Milvus 中对齐
实体/关系 → 写入 Neo4j          ←对应→     在 Neo4j 中查种子节点、扩展关系
chunk_id 关联到 Entity         ←对应→     根据 chunk_id 从 Milvus 回填文本

"""
import json
import logging
import re
from typing import List, Dict, Any, Set
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.config import get_query_config, QueryConfig
from knowledge.processor.query_process.exception import ValidationError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.llm_client import get_llm_client

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
	
	def extract(self, rewritten_query: str)->List[str]:
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
		
	def align(self,entities):
		pass
	

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
		self.log_step(step_name="STEP-2",message="抽取实体名称")
		# 1. 用户问题中抽取实体名称
		entity_extractor = EntityExtractor()
		cleaned_entities = entity_extractor.extract(rewritten_query)
		
		# 2. 基于抽取的实体名称和导入时存储在Milvus中的实体名称进行对齐
		
