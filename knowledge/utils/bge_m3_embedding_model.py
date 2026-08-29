from typing import Optional, List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from dotenv import load_dotenv
import os, logging
from logging import INFO

logger = logging.getLogger(__name__)
logging.basicConfig(level=INFO)
load_dotenv(override=True)

bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None


def get_bge_m3_embedding_model() -> Optional[BGEM3EmbeddingFunction]:
	"""
	获取 BGE-M3 混合嵌入模型（单例模式）。

	从环境变量读取模型配置并初始化 BGEM3EmbeddingFunction 实例，
	首次调用时创建，后续调用直接复用已创建的实例，避免重复加载模型。

	环境变量:
		BGE_M3_PATH: 模型路径或模型名称，默认 "BAAI--bge-m3"
		BGE_DEVICE: 运行设备，如 'cpu' 或 'cuda:0'，默认 "cpu"
		BGE_FP16: 是否启用半精度（fp16），默认 False

	Returns:
		Optional[BGEM3EmbeddingFunction]: 初始化好的 BGE-M3 嵌入模型实例；
		初始化失败时记录错误日志并返回 None。
	"""
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
		return  None


def generate_hybrid_embeddings(
		embedding_model: BGEM3EmbeddingFunction,
		embedding_docs: List[str]
) -> dict|None:
	"""
	为文档列表生成混合嵌入向量（稠密 + 稀疏）。

	使用 BGE-M3 模型对每个文档进行编码，并将结果解析为
	稠密向量（dense）与稀疏向量（sparse）两类，
	便于后续用于混合检索（Hybrid Search）。

	Args:
		embedding_model: BGEM3EmbeddingFunction 嵌入模型实例
		embedding_docs: 待编码的文档文本列表

	Returns:
		dict: 包含 "dense"（稠密向量列表）和 "sparse"（稀疏向量字典列表，
		格式为 {token_id: weight}）两个键的混合嵌入结果
	"""
	
	try:
		doc_embeddings = embedding_model.encode_documents(embedding_docs)
		hybrid_embeddings_result = {
			"dense": [],
			"sparse": []
		}
		
		# 解析稀疏和稠密向量
		for index, chunk in enumerate(embedding_docs):
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
			
			hybrid_embeddings_result["dense"].append(dense_vector)
			hybrid_embeddings_result["sparse"].append(sparse_vector)
		
		return hybrid_embeddings_result
	except Exception as e:
		logger.error(f"BGE-M3生成混合嵌入向量失败: {e}")
		return None
