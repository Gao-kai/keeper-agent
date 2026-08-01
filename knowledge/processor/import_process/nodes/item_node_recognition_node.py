"""
商品名识别节点

通过提前将商品名有关chunk交给LLM生成精确的商品名称后，
调用BGE-M3模型生成稀疏向量和稠密向量存入Milvus向量数据库，
增加用户在询问和商品名相关问题时召回率和精确度。

非流式输出时千问模型必须强制设置enabled-thinks为false
LLM的缓存封装

- response_format=True （开 JSON 模式）
- 在 prompt 里出现 "json" 字样，例如把 system prompt 改成 "请以 JSON 格式输出，不要输出 Markdown 或解释"

稀疏向量（关键词相似检索，但是容易出现苹果和苹果手机混淆的概念）和稠密向量（模糊语义相似检索，但是容易不能精确定位到专有名词）
模型参数BGE-3
encode_queries
encode_documents
dense Numpy数组对象 默认维度1024
sparse 压缩行矩阵  250002

Milvus各种索引的使用
Milvus各种集合的名称
稠密索引COS 余弦相似度
稀疏索引IP 内积

"""
import json
from typing import List, Optional, Tuple, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config, ImportConfig
from knowledge.processor.import_process.exception import ValidationError, ConfigurationError, EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model
from knowledge.utils.llm_client import get_llm_client
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


class ItemNameRecognitionNode(BaseNode):
	name = "item_name_recognition_node"
	
	def process(self, state: ImportGraphState) -> ImportGraphState:
		config: ImportConfig = get_import_config()
		# 1. 参数校验
		file_title, chunks = self.validate_inputs(state)
		
		# 2. 提取商品名称上下文
		item_name_recognition_context = self.build_context(chunks, config.item_name_chunk_k)
		
		# 3. 构建调用LLM大模型识别商品名上下文
		item_name = self.recognize_item_name_by_llm(file_title, item_name_recognition_context, config)
		
		# 4. 调用向量模型生成商品名向量
		dense_vector, sparse_vector = self.embedding_item_name(item_name)
		
		# 5. 存入Milvus向量数据库
		
		# 6. 更新state
		return state
	
	def validate_inputs(self, state: ImportGraphState):
		self.log_step(step_name="STEP-1", message="校验输入参数")
		chunks = state.get("chunks", [])
		file_title = state.get("file_title", "")
		
		if not file_title:
			raise ValidationError(node_name=self.name, message="file_title为空")
		
		if not chunks or not isinstance(chunks, list):
			raise ValidationError(node_name=self.name, message="切分块chunks为空或无效")
		
		self.logger.info(f"文件标题{file_title},共切分为{len(chunks)}个片段")
		
		return file_title, chunks
	
	def build_context(self, chunks: List[dict], top_k: int, max_context_length: int = 2000):
		"""
		取出当前chunks的前top K个chunks 拼接成发送给LLM提取商品名的上下文
		Args:
			chunks: 切片数组
			top_k: 截取前多少个
			max_context_length: 上下文最大长度

		Returns: 前top k个切片的字符串
		"""
		self.log_step(step_name="STEP-2", message="提取商品名称上下文切片")
		if top_k <= 0:
			raise ConfigurationError(node_name=self.name, message="提取商品名称上下文切片数量配置不合法")
		
		context_total = 0
		context_chunks = []
		
		for index, chunk in enumerate(chunks[:top_k]):
			if not isinstance(chunk, dict):
				self.logger.info(f"当前chunk非字典对象，无法继续处理")
				continue
			
			# f"{section_title}\n\n{section_body}",
			chunk_content = chunk.get("content", "")
			
			# 对某超出上下文块直接进行截取
			if len(chunk_content) > max_context_length:
				chunk_content = chunk_content[:max_context_length]
			
			# 加入结果数组
			chunk_content = f"第{index + 1}个切片\n\n{chunk_content}"
			context_chunks.append(chunk_content)
			context_total += len(chunk_content)
			
			# 当前chunk累计总长度超出上下文上限 不收集了 直接中止循环
			if context_total > max_context_length:
				break
		
		# 最后兜底：哪怕数组中收集的chunk长度已经超出上下文上限 最终截取长度也小于max_context_length
		return "\n\n".join(context_chunks)[:max_context_length]
	
	def recognize_item_name_by_llm(self, file_title, item_name_recognition_context, config: ImportConfig):
		self.log_step(step_name="STEP-3", message="调用LLM大模型识别商品名称")
		
		# 1 初始化LLM
		llm_client = get_llm_client(model=config.default_model, temperature=0.1, response_json=False)
		if llm_client is None:
			self.logger.warning("LLM初始化失败，降级为文件名{file_title}")
			return file_title
		
		try:
			# 2 构建提示词模版 ⚠️ 对于SystemMessage、HumanMessage这种不会处理变量占位 所以必须使用元组或者字典
			chat_prompt = ChatPromptTemplate.from_messages([
				("system", ITEM_NAME_SYSTEM_PROMPT),
				("user", ITEM_NAME_USER_PROMPT_TEMPLATE)
			])
			
			final_prompt = chat_prompt.invoke({
				"file_title": file_title,
				"context": item_name_recognition_context
			})
			
			# 3 调用模型
			response = llm_client.invoke(final_prompt)
			item_name = getattr(response, "content", "").strip()
			
			# 4 降级方案
			if not item_name or item_name.upper() == "UNKNOWN":
				self.logger.warning("LLM未能识别商品名称，降级为文件名{file_title}")
				return file_title
			
			# 5 返回结果
			self.logger.info(f"LLM识别商品名称结果为: {item_name}")
			return item_name
		except Exception as e:
			self.logger.warning(f"调用LLM大模型提取商品名称失败:{e}，降级为文件名{file_title}")
			return file_title
	
	def embedding_item_name(self, item_name: str) -> Optional[Tuple[Any, Any]]:
		self.log_step(step_name="STEP-4", message="商品名基于BGE-M3嵌入模型进行嵌入操作")
		try:
			bge_m3_ef = get_bge_m3_embedding_model()
			queries = [item_name]
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
			
			# 打包成字典
			sparse_vector = dict(zip(token_ids, weights))
			
			return dense_vector, sparse_vector
		
		except Exception as e:
			raise EmbeddingError(f"商品名称{item_name}嵌入失败: {e}")


if __name__ == "__main__":
	item_name_recognition_node = ItemNameRecognitionNode()
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks.json", "r",
	          encoding="utf-8") as f:
		chunks_json = f.read()
		chunks_ = json.loads(chunks_json)
	
	item_name_recognition_node.process({
		"chunks": chunks_,
		"file_title": "万用表RS-12的使用"
	})
