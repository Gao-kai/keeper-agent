import json
from dataclasses import dataclass
from typing import Sequence, Dict, List

from pymilvus import DataType, MilvusClient, CollectionSchema

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config
from knowledge.processor.import_process.exception import ValidationError, MilvusError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client import get_milvus_client

"""
建造者设计模式
MilvusSchemaBuilder
MilvusIndexBuilder
MilvusInserterBuilder
专门管理Milvus标量字段类，提取标量字段为类的属性

Milvus的约束：
1. 主键唯一
2. 向量字段基本唯一
3. 标量字段会经常更新
"""

"""
创建Milvus Schema类
"""
class MilvusSchemaBuilder:
	@staticmethod
	def build(milvus_client: MilvusClient, vector_dim: int):
		schema = milvus_client.create_schema(enable_dynamic_field=True)
		
		# 主键字段
		schema.add_field(
			field_name="chunk_id",
			datatype=DataType.INT64,
			is_primary=True,
			auto_id=True
		)
		
		# 添加稠密向量字段
		schema.add_field(
			field_name="dense_vector",
			datatype=DataType.FLOAT_VECTOR,
			dim=vector_dim
		)
		
		# 添加稀疏向量字段
		schema.add_field(
			field_name="sparse_vector",
			datatype=DataType.SPARSE_FLOAT_VECTOR,
		)
		
		# 标量字段
		for field in SCALAR_FIELDS:
			kwargs: Dict[str, int | str | bool] = {
				"field_name": field.file_name,
				"datatype": field.datatype,
			}
			if field.max_length is not None:
				kwargs["max_length"] = field.max_length
			if field.nullable:
				kwargs["nullable"] = True
			
			schema.add_field(**kwargs)
		
		# 返回构建完成的schema
		return schema

"""
创建Milvus 索引Index类
"""
class MilvusIndexBuilder:
	@staticmethod
	def build(milvus_client: MilvusClient):
		index_params = milvus_client.prepare_index_params()
		
		index_params.add_index(
			index_name="chunk_dense_vector_index",
			index_type="IVF_FLAT",
			field_name="dense_vector",
			metric_type="COSINE",
			params={
				"nlist": 64
			}
		)
		index_params.add_index(
			index_name="chunk_sparse_vector_index",
			index_type="SPARSE_INVERTED_INDEX",
			field_name="sparse_vector",
			metric_type="IP"
		)
		
		return index_params

"""
创建Milvus插入数据类
"""
class MilvusInserterBuilder:
	@staticmethod
	def build(milvus_client: MilvusClient, collection_name: str, chunks: List[Dict[str, any]]):
		try:
			milvus_client.load_collection(collection_name=collection_name)
			result = milvus_client.insert(
				collection_name=collection_name,
				data=chunks
			)
			return result
		except Exception as e:
			raise MilvusError(node_name="save_to_milvus_node", message=f"保存到Milvus报错{e}")


"""
标量字段约束类
添加dataclass自动实现init方法、repr方法
"""

@dataclass(frozen=True)
class ScalarFieldSpec:
	file_name: str
	datatype: DataType
	max_length: int = None
	nullable: bool = False

"""
定义全局标量字段列表
"""
SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
	ScalarFieldSpec(file_name="content", datatype=DataType.VARCHAR, max_length=65535),
	ScalarFieldSpec(file_name="file_title", datatype=DataType.VARCHAR, max_length=1024),
	ScalarFieldSpec(file_name="parent_title", datatype=DataType.VARCHAR, max_length=1024),
	ScalarFieldSpec(file_name="item_name", datatype=DataType.VARCHAR, max_length=1024),
	ScalarFieldSpec(file_name="part", datatype=DataType.INT64, nullable=True),
)


class SaveToMilvusNode(BaseNode):
	name = "save_to_milvus_node"
	
	def process(self, state: ImportGraphState) -> ImportGraphState:
		config = get_import_config()
		# 1. 验证参数
		vector_dim, validated_chunks = self.validate_input(state)
		
		# 2. 获取milvus客户端
		self.log_step(step_name="STEP-2", message="获取MILVUS客户端")
		milvus_client = get_milvus_client(uri=config.milvus_url)
		if milvus_client is None:
			raise MilvusError(node_name=self.name, message="获取Milvus客户端失败")
		
		# 3. 构建Schema collection
		collection_name = config.chunks_collection or "keeper_chunks"
		self.ensure_collection(milvus_client, collection_name, vector_dim)
		
		# 4. 存入向量数据库并且回填chunk_id 主键
		ids, chunks = self.insert_milvus(milvus_client, collection_name, validated_chunks)
		
		# 5. 回填ID
		filled_chunks = self.fill_in_primary_key_to_chunk(ids, chunks)
		state["chunks"] = filled_chunks
		
		return state
	
	def validate_input(self, state):
		self.log_step(step_name="STEP-1", message="校验参数")
		chunks = state.get("chunks", [])
		if not chunks:
			raise ValidationError(node_name=self.name, message=f"chunks切片数据为空")
		
		# 确保所有给到向量数据库的都是有向量数据
		validated_chunks = []
		for chunk in chunks:
			if not chunk.get("dense_vector", "") or not chunk.get("sparse_vector"):
				continue
			validated_chunks.append(chunk)
		
		# 获取稠密向量的DIM维度数据
		dim = len(validated_chunks[0].get("dense_vector", []))
		return dim, validated_chunks
	
	def ensure_collection(self, milvus_client: MilvusClient, collection_name, vector_dim):
		self.log_step(step_name="STEP-3", message=f"创建集合{collection_name}")
		
		# 如果该集合已经存在 跳过创建
		if milvus_client.has_collection(collection_name=collection_name):
			return None
		
		# 创建schema
		schema = MilvusSchemaBuilder.build(milvus_client, vector_dim=vector_dim)
		
		# 创建索引
		index_params = MilvusIndexBuilder.build(milvus_client)
		
		# 创建集合
		collection = milvus_client.create_collection(
			collection_name=collection_name,
			schema=schema,
			index_params=index_params
		)
		
		return collection
	
	def insert_milvus(self, milvus_client, collection_name, chunks):
		self.log_step(step_name="STEP-4", message=f"插入数据到Milvus数据库的{collection_name}集合中")
		result = MilvusInserterBuilder.build(milvus_client, collection_name, chunks)
		insert_count = result.get("insert_count", 0)
		ids = result.get("ids", [])
		return ids, chunks
	
	def fill_in_primary_key_to_chunk(self, ids: List[int], chunks: List[Dict[str, any]]):
		self.log_step(step_name="STEP-5", message=f"执行回填 将插入后的主键ID写到chunk中")
		
		if ids and len(ids) == len(chunks):
			for chunk, chunk_id in zip(chunks, ids):
				chunk["chunk_id"] = str(chunk_id)
		else:
			self.logger.warning(
				f"回填 chunk_id 失败: 返回 {len(ids)} 个 ID，"
				f"期望 {len(chunks)} 个"
			)
		
		return chunks


if __name__ == "__main__":
	save_to_milvus_node = SaveToMilvusNode()
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json", "r",
	          encoding="utf-8") as f:
		chunk_contents = json.load(f)
	
	_state = save_to_milvus_node.process({
		"chunks": chunk_contents,
	})
	
	_chunks = _state["chunks"]
	
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json", "w",
	          encoding="utf-8") as f:
		json.dump(_chunks, f, ensure_ascii=False, indent=4)
	print(
		f"插入Milvus数据库后回填的chunks数据保存至:/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json")
