import json

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config, ImportConfig
from knowledge.processor.import_process.exception import ValidationError, EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model


class ChunkEmbeddingNode(BaseNode):
	name = "chunk_embedding_node_name"
	
	def process(self, state: ImportGraphState) -> ImportGraphState:
		
		config = get_import_config()
		
		# 1. 参数校验
		chunks, embedding_batch_size = self.validate_inputs(state, config)
		chunks_length = len(chunks)
		
		# 2. 加载本地模型bge_m3_ef
		self.log_step(step_name="STEP-02", message="加载BGE-M3向量模型")
		try:
			bge_m3_ef = get_bge_m3_embedding_model()
		except Exception as e:
			self.logger.error(f"导入BGE M3嵌入向量失败: {str(e)}")
			raise EmbeddingError(f"初始化 BGE-M3 失败: {e}", node_name=self.name)
		
		# 3. 批量向量化chunk数据 设置固定步长为embedding_batch_size
		embedding_batch_chunks = []
		
		for index in range(0, chunks_length, embedding_batch_size):
			batch_start_index = index
			batch_end_index = index + embedding_batch_size
			batch_chunks = chunks[batch_start_index:batch_end_index]
			
			# 基于BGE-M3模型批量生成稀疏向量和稠密向量
			embedding_batch_chunk = self.process_batch_chunks(bge_m3_ef, batch_chunks, batch_start_index,
			                                                  batch_end_index)
			embedding_batch_chunks.extend(embedding_batch_chunk)
		
		# 4. 更新State Chunks
		self.log_step(step_name="STEP-04", message="更新状态state")
		state["chunks"] = embedding_batch_chunks
		return state
	
	def validate_inputs(self, state: ImportGraphState, config: ImportConfig):
		self.log_step(step_name="STEP-01", message="校验输入参数")
		
		chunks = state.get("chunks", [])
		if not chunks or not isinstance(chunks, list):
			raise ValidationError(node_name=self.name, message=f"切片chunks为空或不合法")
		
		embedding_batch_size = config.embedding_batch_size
		if not embedding_batch_size or embedding_batch_size <= 0:
			raise ValidationError(node_name=self.name,
			                      message=f"批量切片个数参数embedding_batch_size为{embedding_batch_size}不合法")
		
		return chunks, embedding_batch_size
	
	def process_batch_chunks(self, bge_m3_ef, batch_chunks, batch_start_index, batch_end_index):
		self.log_step(step_name="STEP-03", message="批量处理切片向量化数据")
		if not batch_chunks:
			return batch_chunks or []
		
		try:
			# 构建本批次要嵌入的文档列表
			docs = []
			for chunk in batch_chunks:
				item_name = chunk.get("item_name")
				item_content = chunk.get("content")
				doc_content = f"{item_name}\n\n{item_content}"
				docs.append(doc_content)
			
			# docs执行批量向量化
			doc_embeddings = bge_m3_ef.encode_documents(docs)
			if not doc_embeddings:
				self.logger.warning(f"批次 {batch_start_index + 1}-{batch_end_index} 未能生成向量")
				return batch_chunks
			
			# chunk新增向量属性
			final_chunks = []
			for index, chunk in enumerate(batch_chunks):
				# 获取稠密向量dense
				dense_vector = doc_embeddings['dense'][index].tolist()
				
				# 获取稀疏向量CSR
				sparse_matrix = doc_embeddings["sparse"]
				
				# 获取第 i 句话非零元素的起止索引
				start_idx = sparse_matrix.indptr[index]
				end_idx = sparse_matrix.indptr[index + 1]
				
				# 提取对应的 Token IDs 和 权重
				token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
				weights = sparse_matrix.data[start_idx:end_idx].tolist()
				
				# 打包成字典 {tokenId:weight}
				sparse_vector = dict(zip(token_ids, weights))
				
				# 为chunk对象新增稀疏和稠密向量属性
				chunk["dense_vector"] = dense_vector
				chunk["sparse_vector"] = sparse_vector
				
				final_chunks.append(chunk)
			
			self.logger.info(
				f"成功处理批次 {batch_start_index + 1}-{batch_end_index}"
			)
			return final_chunks
		
		except Exception as e:
			self.logger.error(
				f"批次 {batch_start_index + 1}-{batch_end_index} 处理失败: {e}"
			)
			return batch_chunks


if __name__ == "__main__":
	chunk_embedding_node = ChunkEmbeddingNode()
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json", "r",
	          encoding="utf-8") as f:
		chunk_contents = json.load(f)
	
	_state = chunk_embedding_node.process({
		"chunks": chunk_contents,
		"file_title": "万用表RS-12的使用"
	})
	_chunks = _state["chunks"]
	
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json", "w",
	          encoding="utf-8") as f:
		json.dump(_chunks, f, ensure_ascii=False, indent=4)
	print(
		f"切片数据向量化后的chunks数据保存至:/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/chunks_vector.json")
