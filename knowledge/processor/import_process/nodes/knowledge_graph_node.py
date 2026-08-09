"""
混合检索 向量数据库 + 图数据库

1. 基于每一个chunk内容抽取实体和实体之间的关系
	- 手动提取 无脏数据 耗时但是数据基本不用清洗
	- LLM自动提取 必须借助于提示词模版来约束大模型提取实体的名称以及实体类型的范围（白名单机制）
2. 清洗LLM模型提取的实体数据
	- JSON代码围栏
	- 实体名称清洗
	- 实体关系清洗
3. 双写之写入向量数据库
	- 将清洗后的实体名称向量化后写入向量数据库
	- 用户的问题中如果有和实体语义模糊匹配的 需要先从向量数据库中匹配到精确的实体名称【实体名对齐】
4. 双写之写入图谱数据库
	- 将抽取到的实体和关联字段chunk_id以及实体之间的关系全部存入到图谱数据库
	
TODO: 最佳实践是构建“混合流水线
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from neo4j import Driver
from pymilvus import MilvusClient

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config
from knowledge.processor.import_process.exception import ValidationError, LLMError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.milvus_client import get_milvus_client
from knowledge.utils.neo4j_client import get_neo4j_client


@dataclass
class ProcessingState:
	"""
	记录处理过程中的状态和节点信息 用于记录日志和监控
	"""
	
	total_chunks: int = 0
	processed_chunks: int = 0
	failed_chunks: int = 0
	
	total_entities: int = 0
	total_relations: int = 0
	errors: List[str] = field(default_factory=list)
	
	def summary(self):
		return (
			f"全部切片数量：{self.total_chunks}",
			f"处理完成切片数量：{self.processed_chunks}/{self.total_chunks}",
			f"处理失败切片数量：{self.failed_chunks}/{self.total_chunks}",
			f"全部实体节点数量：{self.total_entities}",
			f"全部实体关系数量：{self.total_relations}"
		)


class KnowledgeGraphNode(BaseNode):
	name = "extract_entity_node"
	
	def process(self, state: ImportGraphState) -> ImportGraphState:
		config = get_import_config()
		item_name = state.get("item_name", "")
		# 1. 输入数据校验
		validated_chunks = self.validate_inputs(state)
		
		# 2. 构建日志记录
		process_state = ProcessingState(total_chunks=len(validated_chunks))
		
		# 3. 清除已经存在数据
		# 删除Milvus中存储实体名字的记录
		# 删除neo4j的整个库下的所有节点和信息
		milvus_client = get_milvus_client()
		neo4j_client = get_neo4j_client()
		self.clear_exist_data(milvus_client, neo4j_client, item_name, process_state)
		
		# 3. 批量处理遍历chunks【串行】TODO: 多线程版本
		self.process_all_chunks(validated_chunks, milvus_client, neo4j_client, process_state)
		
		return state
	
	def validate_inputs(self, state):
		"""
		保证后续处理过程中参数校验都合法
		Args:
			state:

		Returns:

		"""
		chunks = state.get("chunks", [])
		if not chunks:
			raise ValidationError(node_name=self.name, message="待构建知识图谱的chunks不存在")
		
		item_name = state.get("item_name", "").strip()
		
		validated_chunks = []
		
		for index, chunk in enumerate(chunks):
			if not isinstance(chunk, dict):
				self.logger.warning(f"第{index + 1}个chunk为非字典类型，该chunk跳过图谱构建")
				continue
			
			raw_id = chunk.get("chunk_id", "")
			chunk_id = str(raw_id).strip() if raw_id is not None else f"temp_chunk_{index + 1}"
			
			content = chunk.get("content", "").strip()
			if not content:
				self.logger.warning(f"第{index + 1}个chunk无content，该chunk跳过图谱构建")
				continue
			
			chunk_item_name = chunk.get("item_name", "").strip() or item_name
			if not chunk_item_name:
				self.logger.warning(f"第{index + 1}个chunk无商品名称item_name属性，该chunk跳过图谱构建")
				continue
			
			# 更新chunk字段
			chunk["chunk_id"] = chunk_id
			chunk["content"] = content
			chunk["item_name"] = chunk_item_name
			
			validated_chunks.append(chunk)
		
		if not validated_chunks:
			raise ValidationError(node_name=self.name, message="校验后待处理的切片chunks为空")
		
		return validated_chunks
	
	def clear_exist_data(
			self,
			milvus_client: MilvusClient,
			neo4j_client: Driver,
			item_name: str,
			process_state: ProcessingState
	):
		pass
	
	def process_all_chunks(
			self,
			validated_chunks: List[Dict],
			milvus_client: MilvusClient,
			neo4j_client: Driver,
			process_state: ProcessingState
	):
		
		for index, chunk in enumerate(validated_chunks):
			chunk_id = chunk.get("chunk_id")
			
			try:
				
				entities_count, relations_count = self.process_single_chunk(chunk, milvus_client, neo4j_client)
				process_state.processed_chunks += 1
				process_state.total_entities += entities_count
				process_state.total_relations += relations_count
				self.logger.info(f"成功处理{chunk_id}图谱构建")
			except Exception as e:
				process_state.failed_chunks += 1
				process_state.errors.append(str(e))
				self.logger.error(f"处理{chunk_id}图谱构建失败")
	
	def process_single_chunk(self, chunk: Dict[str, Any], milvus_client: MilvusClient, neo4j_client: Driver):
		
		# 1. 调用LLM模型提取当前chunk的实体和关系[重试机制]
		llm_reponse = self.extract_graph_with_retry(chunk)
	
	# 2. 解析并清洗LLM返回结果
	
	# 3. 将清洗后的实体名字写入至Milvus向量数据库【目的：用户提问时先从向量数据库中找到精确实体名称】
	
	# 4. 将清洗后的实体名字和关系写入到Neo4j图谱中 【目的：混合检索阶段可以查询到问题中的实体关联关系】
	def extract_graph_with_retry(self, chunk: Dict[str, Any]):
		"""
		提供自动重试机制的模型调用
		Args:
			chunk:

		Returns:

		"""
		chunk_id = chunk.get("chunk_id")
		item_name = chunk.get("item_name")
		content = chunk.get("content")
		
		llm_client = get_llm_client(response_json=True)
		if llm_client is None:
			raise LLMError(node_name=self.name, message="初始化LLM客户端失败")
		
		config = get_import_config()
		llm_errors = []
		messages = [
			SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
			HumanMessage(content=f"请处理以下文本:\n\n {content}")
		]
		for attempt in range(1, config.max_call_llm_attempt_count + 1):
			try:
				response = llm_client.invoke(
					messages
				)
				
				extract_result = getattr(response, "content", "").strip()
				
				if extract_result:
					return extract_result
			except Exception as e:
				llm_errors.append(str(e))
				if attempt < config.max_call_llm_attempt_count:
					# 1 - 2 -4 -8 设置每次等待时间翻倍
					delay = 1 * (2 ** (attempt - 1))
					self.logger.info(f"调用LLM提取实体失败:开启第{attempt}次重试，等待{delay}秒")
					time.sleep(delay)
			
			self.logger.error(f"已经进行{config.max_call_llm_attempt_count}次重试")
			self.logger.error(f"错误信息为:{'\n\n'.join(llm_errors)}")
		
		return ""
