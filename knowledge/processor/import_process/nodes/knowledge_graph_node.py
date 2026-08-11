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
TODO: 不一定需要把所有实体都提取出来 我们的目的是找到原始的chunk块
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set

from langchain_core.messages import SystemMessage, HumanMessage
from neo4j import Driver
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config, ImportConfig
from knowledge.processor.import_process.exception import ValidationError, LLMError, MilvusError, EmbeddingError, \
	Neo4jError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.milvus_client import get_milvus_client
from knowledge.utils.neo4j_client import get_neo4j_client

# 实体类型白名单（SET）
ALLOWED_ENTITY_LABEL_TYPES = {
	"Device",
	"Part",
	"Operation",
	"Step",
	"Warning",
	"Condition",
	"Tool"
}

# 实体关系类型白名单（SET）
ALLOWED_RELATION_TYPES = {
	"HAS_OPERATION",
	"HAS_PART",
	"HAS_STEP",
	"USES_TOOL",
	"HAS_WARNING",
	"NEXT_STEP",
	"AFFECTS",
	"REQUIRES",
	"MENTIONED_IN",
	"RELATED_TO",
}

# 默认关系类型
DEFAULT_RELATION_TYPE = "RELATED_TO"


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
		
		milvus_client = get_milvus_client()
		if milvus_client is None:
			raise MilvusError(node_name=self.name, message="Milvus客户端连接失败")
		
		neo4j_client = get_neo4j_client()
		if neo4j_client is None:
			raise Neo4jError(node_name=self.name, message="Neo4j客户端连接失败")
		
		# 3. 清除（保证幂等性 防止出现重复创建）
		self.clear_exist_data(milvus_client, neo4j_client, item_name, config)
		
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
			config: ImportConfig
	):
		"""
		删除集合collection_name下面：
		1. 所有item_name为item_name的实体节点
		2. 所有item_name为item_name的实体和关系节点
		Args:
			milvus_client:
			neo4j_client:
			item_name:
			config:

		Returns:

		"""
		collection_name = config.entity_name_collection
		if not collection_name:
			raise MilvusError(node_name=self.name, message=f"Milvus 集合{collection_name}不存在")
		
		# 删除Milvus中存储实体名字的记录
		try:
			if milvus_client.has_collection(collection_name):
				milvus_client.delete(
					collection_name=collection_name,
					filter=f"item_name == '{item_name}'"
				)
				self.logger.info(f"已清空所有商品名为{item_name}的实体节点")
		except Exception as e:
			raise MilvusError(node_name=self.name, message=f"Milvus删除集合失败:{e}")
		
		# 删除neo4j的所有item_name为item_name的实体和关系节点
		try:
			neo4j_client.execute_query(
				query_="""
				MERGE (n:ENTITY {item_name:$item_name})
				DETACH DELETE n
				""",
				parameters_={
					"item_name": item_name
				},
				database_=config.neo4j_database
			)
		except Exception as e:
			raise Neo4jError(node_name=self.name, message=f"Neo4j删除{item_name}有关节点和关系失败:{e}")
	
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
				entities, relations = self.process_single_chunk(chunk, milvus_client, neo4j_client)
				process_state.processed_chunks += 1
				process_state.total_entities += len(entities)
				process_state.total_relations += len(relations)
				self.logger.info(f"成功处理{chunk_id}图谱构建")
			except Exception as e:
				process_state.failed_chunks += 1
				process_state.errors.append(str(e))
				self.logger.error(f"处理{chunk_id}图谱构建失败:{e}")
	
	def process_single_chunk(self, chunk: Dict[str, Any], milvus_client: MilvusClient, neo4j_client: Driver):
		chunk_id = chunk.get("chunk_id", "")
		
		config = get_import_config()
		
		# 1. 调用LLM模型提取当前chunk的实体和关系[重试机制]
		llm_response = self.extract_graph_with_retry(chunk, config)
		
		if not llm_response:
			return None
		
		# 2. 解析并清洗LLM返回结果
		graph_data: Dict[str, List] = self.parse_and_clean_llm_result(llm_response, config)
		
		if not graph_data:
			return None
		
		entities = graph_data.get("entities", [])
		relations = graph_data.get("relations", [])
		
		self.logger.info(f"切片 {chunk_id}的切片: "
		                 f"提取到 {len(entities)} 个实体\n"
		                 f"提取到 {len(relations)} 条关系")
		
		# 4. 将清洗后的实体名字写入至Milvus向量数据库【目的：用户提问时先从向量数据库中找到精确实体名称】
		self.save_entity_names_to_milvus(entities, config, chunk, milvus_client)
		
		# 5. 将清洗后的实体名字和关系写入到Neo4j图谱中 【目的：混合检索阶段可以查询到问题中的实体关联关系】
		self.save_graph_data_to_neo4j(entities, relations, config, chunk, neo4j_client)
		
		# 6. 返回处理完成的节点数据
		return entities, relations
	
	def extract_graph_with_retry(self, chunk: Dict[str, Any], config: ImportConfig):
		"""
		TODO: 错误原因、响应为空作为AI Message进行重试的消息列表
		提供自动重试机制的模型调用

		Args:
			chunk:
			config

		Returns:

		"""
		chunk_id = chunk.get("chunk_id")
		item_name = chunk.get("item_name")
		content = chunk.get("content")
		
		llm_client = get_llm_client(response_json=True)
		if llm_client is None:
			raise LLMError(node_name=self.name, message="初始化LLM客户端失败")
		
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
	
	def parse_and_clean_llm_result(self, llm_response: str, config) -> Dict[str, Any]:
		"""
		对LLM返回的内容进行校验和清洗：
		1. 可能包含markdown的json围栏
		2. 返回的JSON反序列化为字典失败
		3. 清洗实体列表
		4. 清洗关系列表

		Args:
			llm_response:
			config:

		Returns:

		"""
		if not llm_response:
			return {
				"entities": [],
				"relations": []
			}
		
		# 消除可能返回的```json```代码围栏
		cleaned_text = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
		cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
		
		# json反序列化可能失败
		try:
			data: Dict[str, Any] = json.loads(cleaned_text)
		except json.JSONDecodeError as e:
			self.logger.error(f"JSON反序列化失败: {e}")
			return {
				"entities": [],
				"relations": []
			}
		
		# 清洗实体列表
		entities = data.get("entities", [])
		cleaned_entities = self.clean_entities_from_llm_result(entities, config)
		# Set结构天然去重特性获取去重后的实体名称构成的集合
		cleaned_entity_names: Set[str] = {item["name"] for item in cleaned_entities}
		
		# 清洗关系列表
		relations = data.get("relations", [])
		cleaned_relations = self.clean_relations_from_llm_result(relations, cleaned_entity_names, config)
		
		return {
			"entities": cleaned_entities,
			"relations": cleaned_relations
		}
	
	def clean_entities_from_llm_result(self, entities: List[Dict[str, Any]], config: ImportConfig):
		"""
		
		清洗实体名称：
		2. 实体名称为空
		3. 实体名称超出长度
		4. 同一名称同一Label的实体重复提取
		5. 实体标签Label不存在于白名单中
		
		Args:
			entities:
			config

		Returns:

		"""
		
		if not entities:
			return []
		
		cleaned_entities: List[Dict[str, Any]] = []
		seen: Set[Tuple] = set()
		
		for entity in entities:
			
			# 为空判断
			entity_name = entity.get("name", "")
			entity_label = entity.get("label", "")
			if not entity_name or not entity_label:
				continue
			
			# 超出长度截取
			if len(entity_name) > config.max_entity_name_length:
				entity_name = entity_name[:config.max_entity_name_length + 1]
			
			# 去重
			entity_unique_key = (entity_name, entity_label)
			
			if entity_unique_key in seen:
				continue
			seen.add(entity_unique_key)
			
			# 实体Label不在白名单
			if entity_label not in ALLOWED_ENTITY_LABEL_TYPES:
				continue
			
			cleaned_entity = {
				"name": entity_name,
				"label": entity_label,
			}
			
			# 是否包含描述
			entity_desc = entity.get("description", "")
			if not entity_desc:
				cleaned_entity["description"] = entity_desc
			
			cleaned_entities.append(cleaned_entity)
		
		self.logger.info(f"清洗实体完成：{len(cleaned_entities)}")
		
		return cleaned_entities
	
	def clean_relations_from_llm_result(
			self,
			relations: List[Dict[str, Any]],
			cleaned_entity_names: Set[str],
			config: ImportConfig
	):
		"""
		清洗关系列表
		1. 为空
		2. 长度超出
		3. 悬空关系
		
		Args:
			relations:
			cleaned_entity_names:
			config:

		Returns:

		"""
		
		if not relations:
			return []
		
		cleaned_relations: List[Dict[str, Any]] = []
		
		for relation in relations:
			
			# 为空判断
			head_entity_name = relation.get("head", "")
			tail_entity_name = relation.get("tail", "")
			relation_type = relation.get("type", "")
			
			if not head_entity_name or not tail_entity_name:
				continue
			
			# 超出长度截取
			if len(head_entity_name) > config.max_entity_name_length:
				head_entity_name = head_entity_name[:config.max_entity_name_length + 1]
			
			if len(tail_entity_name) > config.max_entity_name_length:
				tail_entity_name = tail_entity_name[:config.max_entity_name_length + 1]
			
			# 关系类型不在白名单 此时可以给默认关系
			if relation_type not in ALLOWED_RELATION_TYPES:
				relation_type = DEFAULT_RELATION_TYPE
			
			# 悬空关系：一个关系的头尾节点的名称都不在cleaned_entity_names里面
			if head_entity_name not in cleaned_entity_names or tail_entity_name not in cleaned_entity_names:
				continue
			
			cleaned_relation = {
				"head": head_entity_name,
				"tail": tail_entity_name,
				"type": relation_type,
			}
			
			cleaned_relations.append(cleaned_relation)
		
		self.logger.info(f"清洗关系完成：{len(cleaned_relations)}")
		
		return cleaned_relations
	
	def save_entity_names_to_milvus(self, entities: List[Dict[str, Any]], config: ImportConfig, chunk: Dict[str, Any],
	                                milvus_client: MilvusClient):
		"""
		存储所有实体名称到Milvus向量数据库
		Args:
			entities:
			config:
			chunk:
			milvus_client:

		Returns:

		"""
		try:
			# 1. 构建Collection
			collection_name = config.entity_name_collection
			if not milvus_client.has_collection(collection_name=collection_name):
				self.create_entity_name_collection(milvus_client, collection_name)
			
			# 2. 获取BGE-M3本地向量模型
			bge_m3_ef = get_bge_m3_embedding_model()
			
			# 3. 生成数据
			records = []
			chunk_content = chunk.get("content", "")
			chunk_id = chunk.get("chunk_id", "")
			item_name = chunk.get("item_name", "")
			
			# 4. 对entities进行去重
			unique_entity_names: Set[str] = {entity["name"] for entity in entities}
			for entity_name in unique_entity_names:
				# 1. 使用BGE-M3本地向量模型将所有实体名称转化为稀疏和稠密向量
				dense_vector, sparse_vector = self.embedding_entity_name(entity_name, bge_m3_ef)
				
				# 2. 填充
				data = {
					"entity_name": entity_name,
					"source_chunk_id": chunk_id,
					"context": chunk_content[:200],
					"item_name": item_name,
				}
				
				if dense_vector is not None:
					data["entity_name_dense_vector"] = dense_vector
				
				if sparse_vector is not None:
					data["entity_name_sparse_vector"] = sparse_vector
				
				records.append(data)
			
			# 5. 插入集合(data可以是一条数据也可以数据组成的列表)
			inserted_response = milvus_client.insert(
				collection_name=collection_name,
				data=records
			)
			
			# 6. 插入数据后，强制刷新数据到磁盘
			milvus_client.flush(collection_name=collection_name)
			self.logger.info(f"实体名称向量插入到集合{collection_name}成功")
		except Exception as e:
			self.logger.error(f"存储实体名称向量到Milvus数据库失败:{e}")
	
	def embedding_entity_name(self, entity_name: str, bge_m3_ef):
		self.logger.info(f"当前处理的实体名称为: {entity_name}")
		try:
			
			queries = [entity_name]
			query_embeddings = bge_m3_ef.encode_queries(queries)
			
			# 获取稠密向量dense
			dense_vector = query_embeddings['dense'][0].tolist()
			
			# 获取稀疏向量CSR
			sparse_matrix = query_embeddings["sparse"]
			# 获取第 i 句话非零元素的起止索引
			start_idx = sparse_matrix.indptr[0]
			end_idx = sparse_matrix.indptr[1]
			
			# 提取对应的 Token IDs 和 权重
			token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
			weights = sparse_matrix.data[start_idx:end_idx].tolist()
			
			print(f"稀疏向量的非零元素的索引列表：{start_idx}-{end_idx}")
			print(f"稀疏向量的非零元素的权重列表：{weights}")
			print(f"稀疏向量的非零元素的TokenID列表：{token_ids}")
			
			# 打包成字典 {tokenId:weight}
			sparse_vector = dict(zip(token_ids, weights))
			
			return dense_vector, sparse_vector
		
		except Exception as e:
			raise EmbeddingError(node_name=self.name,message=f"实体名称{entity_name}嵌入失败: {e}")
	
	def create_entity_name_collection(self, milvus_client: MilvusClient, collection_name: str):
		# 创建Schema
		schema = milvus_client.create_schema(enable_dynamic_field=True)
		
		# 添加主键字段
		schema.add_field(
			field_name="entity_id",
			datatype=DataType.INT64,
			is_primary=True,
			auto_id=True
		)
		
		# 添加标量字段
		schema.add_field(
			field_name="entity_name",
			datatype=DataType.VARCHAR,
			max_length=1024
		)
		schema.add_field(
			field_name="source_chunk_id",
			datatype=DataType.VARCHAR,
			max_length=1024
		)
		schema.add_field(
			field_name="context",
			datatype=DataType.VARCHAR,
			max_length=1024
		)
		schema.add_field(
			field_name="item_name",
			datatype=DataType.VARCHAR,
			max_length=1024
		)
		
		# 添加稠密向量字段
		schema.add_field(
			field_name="entity_name_dense_vector",
			datatype=DataType.FLOAT_VECTOR,
			dim=1024
		)
		
		# 添加稀疏向量字段
		schema.add_field(
			field_name="entity_name_sparse_vector",
			datatype=DataType.SPARSE_FLOAT_VECTOR,
		)
		
		# 添加集合索引(稀疏向量&稠密向量)
		index_params = milvus_client.prepare_index_params()
		index_params.add_index(
			index_name="entity_name_dense_vector_index",
			index_type="IVF_FLAT",
			field_name="entity_name_dense_vector",
			metric_type="COSINE",
			params={
				"nlist": 64
			}
		)
		index_params.add_index(
			index_name="entity_name_sparse_vector_index",
			index_type="SPARSE_INVERTED_INDEX",
			field_name="entity_name_sparse_vector",
			metric_type="IP"
		)
		
		collection = milvus_client.create_collection(
			collection_name=collection_name,
			index_params=index_params,
			schema=schema
		)
		
		self.logger.info(f"创建实体名称集合向量成功，集合名称：{collection_name}")
		return collection
	
	def save_graph_data_to_neo4j(self, entities, relations, config: ImportConfig, chunk: Dict[str, Any],neo4j_driver):
		"""
		1. 新建Chunk节点 类型为Chunk 属性为chunk_id和item_name
		2. 新建实体节点 类型为（通用类型）Entity和LLM返回的单独类型label 属性为：
			- 实体名称
			- 实体描述
			- 源chunk的id
			- item_name商品名
		3. 新建实体和实体之间的关系（基于大模型返回）
		
		查询关系：
		用户自然语言提问，从中提取到实体名称
		拿着实体名称去向量数据库检索最相似实体名称，确定实体名称完全匹配，避免语义相近但是不相等的问题
		拿着确定实体名称在图数据库进行检索，检索到对应的chunk块
		基于chunk id去向量数据库找到最相似的chunk后返回
		
		Args:
			entities:
			relations:
			config:
			chunk:
			neo4j_driver:

		Returns:

		"""
		if not entities or not relations:
			self.logger.error("存入Neo4j图数据库的实体或关系数据不存在")
			return
		
		# 获取neo4j客户端
		database = config.neo4j_database
		chunk_id = chunk.get("chunk_id", ""),
		item_name = chunk.get("item_name", "")
		
		# 新建chunk节点
		neo4j_driver.execute_query(
			query_="""
			MERGE (c:CHUNK {chunk_id:$chunk_id,item_name:$item_name})
			""",
			parameters_={
				"chunk_id": chunk_id,
				"item_name": item_name
			},
			database_=database
		)
		
		for entity in entities:
			name = entity.get("name", "")
			label = entity.get("label", "")
			description = entity.get("description", "")
			
			# 新建实体节点
			neo4j_driver.execute_query(
				# python中f-string原本会将{}中的内容保留用于嵌入遍历，因此{{}}就是在字符串中保留一个单花括号{}
				query_=f"""
				MERGE (n:ENTITY {{name:$name,item_name:$item_name}})
				ON CREATE SET
					n.source_chunk_id = $chunk_id,
					n.description = $description
				ON MATCH SET
					n.description = CASE
						WHEN $description <> "" THEN $description
						ELSE coalesce(n.description,"")
					END
				SET n:`{label}`
				""",
				parameters_={
					"name": name,
					"description": description,
					"chunk_id": chunk_id,
					"item_name": item_name
				},
				database_=database
			)
			
			# 新建当前ENTITY节点和CHUNK节点的关系 关系类型：MENTION——IN
			neo4j_driver.execute_query(
				query_="""
				MATCH (c:CHUNK {chunk_id:$chunk_id,item_name:$item_name})
				MATCH (n:ENTITY {name:$name,item_name:$item_name})
				MERGE (n)-[:MENTIONED_IN]->(c)
				""",
				parameters_={
					"name": name,
					"chunk_id": chunk_id,
					"item_name": item_name
				},
				database_=database
			)
		
		# 新建实体ENTITY节点与实体ENTITY节点之间关联关系
		for relation in relations:
			head_name = relation.get("head", "")
			tail_name = relation.get("tail", "")
			relation_type = relation.get("type", "")
			neo4j_driver.execute_query(
				query_=f"""
				MATCH (head:ENTITY {{name:$head_name,item_name:$item_name}})
				MATCH (tail:ENTITY {{name:$tail_name,item_name:$item_name}})
				MERGE (head)-[:{relation_type}]->(tail)
				""",
				parameters_={
					"head_name": head_name,
					"tail_name": tail_name,
					"item_name":item_name
				},
				database_=database
			)


if __name__ == "__main__":
	knowledgeGraphNode = KnowledgeGraphNode()
	result = knowledgeGraphNode.process({
		"chunks": [
			{
				"content": "## 电池测试\n\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n\n\n- 【9V 电池:】：良好为>8.2V，较弱为7.2 至 8.2V，坏的为<7.2V。\n- 【1.5V 电池:】：良好为>1.35V，较弱为1.22 至 1.35V，坏的为<1.22V。",
				"parent_title": "万用表RS-12的使用",
				"file_title": "万用表RS-12的使用",
				"item_name": "RS-12 数字万用表",
				"chunk_id": "468082957058058874"
			}
		],
		"item_name": "RS-12 数字万用表"
	})
	print(result)
