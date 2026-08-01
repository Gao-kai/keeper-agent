from typing import Optional

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from dotenv import load_dotenv
import os, logging
from logging import INFO

logger = logging.getLogger(__name__)
logging.basicConfig(level=INFO)
load_dotenv(override=True)

bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None


def get_bge_m3_embedding_model():
	try:
		global bge_m3_ef
		
		if bge_m3_ef is not None and isinstance(bge_m3_ef, BGEM3EmbeddingFunction):
			return bge_m3_ef
		
		# 防止字符串中斜杠等符号被转义
		model_name = os.path.normpath(os.getenv("BGE_M3_PATH", "BAAI--bge-m3"))
		model_device = os.getenv("BGE_DEVICE", "cpu")
		bge_fp16_str = os.getenv("BGE_FP16", False)
		
		# os env读取的值是字符串 需要转化为模型需要参数布尔值
		use_fp16 = bge_fp16_str.lower() in ("true", "1", "yes")
		
		bge_m3_ef = BGEM3EmbeddingFunction(
			model_name=model_name,  # Specify the model name
			device=model_device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
			use_fp16=use_fp16  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
		)
		
		return bge_m3_ef
	except Exception as e:
		logger.error(f"初始化BGE-M3向量嵌入模型失败: {e}")


if __name__ == "__main__":
	bge_m3_ef = get_bge_m3_embedding_model()
	queries = ["我喜欢苹果手机"]
	query_embeddings = bge_m3_ef.encode_queries(queries)
	print(f"query_embeddings:{query_embeddings}")
	print(f"稠密向量：{query_embeddings.get("dense", [])[0].tolist()}")
	
	# 获取稀疏向量CSR
	sparse_matrix = query_embeddings["sparse"]
	
	sparse_vectors = []
	
	# 遍历每一句话
	for i in range(len(queries)):
		# 获取第 i 句话非零元素的起止索引
		start_idx = sparse_matrix.indptr[i]
		end_idx = sparse_matrix.indptr[i + 1]
		
		# 提取对应的 Token IDs 和 权重
		token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()
		weights = sparse_matrix.data[start_idx:end_idx].tolist()
		
		print(f"稀疏向量的非零元素的索引列表：{start_idx}-{end_idx}")
		
		print(f"稀疏向量的非零元素的权重列表：{weights}")
		
		print(f"稀疏向量的非零元素的TokenID列表：{token_ids}")
		
		# 打包成字典
		sparse_vector = dict(zip(token_ids, weights))
		sparse_vectors.append(sparse_vector)
	
	print(f"稀疏向量列表：{sparse_vectors}")
