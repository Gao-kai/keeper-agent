# 创建Milvus客户端
from typing import Optional, List

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from logging import INFO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=INFO)
load_dotenv(override=True)

# Milvus 连接地址
# MILVUS_URL = http://localhost:19530
# 知识库切片集合名
# CHUNKS_COLLECTION = keeper_chunks
# 实体名称集合名
# ENTITY_NAME_COLLECTION = keeper_graph_entity_names
# 商品名称集合名
# ITEM_NAME_COLLECTION = keeper_item_names
# 相似度度量方式
# MILVUS_METRIC_TYPE = COSINE
# 最小余弦相似度阈值
# MILVUS_MIN_COSINE_SCORE = 0.75

# 全局单例的 Milvus 客户端
milvus_client: Optional[MilvusClient] = None


def get_milvus_client(uri: str = "") -> Optional[MilvusClient]:
	"""
	获取 Milvus 客户端（单例模式）。

	首次调用时根据 uri 参数或环境变量 MILVUS_URL 创建 MilvusClient 实例，
	后续调用直接复用已创建的实例，避免重复建立连接。

	Args:
		uri: Milvus 连接地址，优先使用该参数；为空时读取环境变量 MILVUS_URL

	Returns:
		Optional[MilvusClient]: Milvus 客户端实例；创建失败时记录错误日志并返回 None
	"""
	try:
		global milvus_client
		
		if milvus_client is not None and isinstance(milvus_client, MilvusClient):
			return milvus_client
		
		milvus_server_uri = uri or os.getenv("MILVUS_URL")
		
		milvus_client = MilvusClient(
			uri=milvus_server_uri
		)
		
		return milvus_client
	except Exception as e:
		logger.error(f"创建Milvus客户端报错: {e}")
		return None


def create_hybrid_search_request(
		dense_vector,
		sparse_vector,
		dense_req_field_name="dense_vector",
		sparse_req_field_name="sparse_vector",
		dense_params=None,
		sparse_params=None,
		expr=None,
		expr_params=None,
		limit=10
) -> List[AnnSearchRequest]:
	"""
	构造混合检索请求（稠密向量检索器 + 稀疏向量检索器）。

	Args:
		dense_vector: 稠密向量，由 BGE-M3 生成的 dense 向量
		sparse_vector: 稀疏向量，由 BGE-M3 生成的 sparse 向量（格式为 {token_id: weight}）
		dense_req_field_name: 稠密向量字段名称
		sparse_req_field_name: 稀疏向量字段名称
		dense_params: 稠密检索参数，如 {"metric_type": "COSINE"}；默认使用 COSINE 度量
		sparse_params: 稀疏检索参数，如 {"metric_type": "IP"}；默认使用 IP 度量
		expr: 标量字段过滤表达式（类似 SQL 的 WHERE 条件）；None 表示不过滤
		expr_params: 标量字段过滤表达式参数
		limit: 每个检索器返回的结果数量上限，默认 10

	Returns:
		List[AnnSearchRequest]: 包含稠密与稀疏两个检索器的请求列表
	"""
	if dense_params is None:
		dense_params = {
			"metric_type": "COSINE"
		}
	
	if sparse_params is None:
		sparse_params = {
			"metric_type": "IP"
		}
	
	# 稠密向量检索器
	dense_req = AnnSearchRequest(
		data=[dense_vector],
		anns_field=dense_req_field_name,
		param=dense_params,
		expr=expr,
		expr_params=expr_params,
		limit=limit
	)
	
	# 稀疏向量检索器
	sparse_req = AnnSearchRequest(
		data=[sparse_vector],
		anns_field=sparse_req_field_name,
		param=sparse_params,
		expr=expr,
		expr_params=expr_params,
		limit=limit
	)
	
	# 这里返回的顺序很重要将直接决定了在执行混合检索的时候ranker_weights数组中的参数对应关系
	# ranker_weights为[0.7.0.3] 意思就是稠密向量检索权重dense_req占比0.7 稀疏向量占比0.3
	return [dense_req, sparse_req]


def execute_hybrid_search(
		milvus_client: MilvusClient,
		limit: int,
		reqs: List[AnnSearchRequest],
		collection_name: str,
		output_fields: List[str],
		ranker_weights: List[float],
		norm_score: bool = True):
	"""
	执行混合检索并返回结果。

	使用加权排序器（WeightedRanker）融合稠密与稀疏两个检索器的得分，
	再对指定集合执行 hybrid_search，输出相关标量字段。

	Args:
		milvus_client: Milvus 客户端实例
		limit: 返回的结果数量上限
		reqs: 检索请求列表，由 create_hybrid_search_request 生成
		collection_name: 目标集合名称
		output_fields: 需要返回的标量字段列表；为 None 时默认返回 ["text", "id"]
		ranker_weights: 各检索器的权重列表，如 [0.5, 0.5]，长度需与检索器数量一致
		norm_score: 是否对融合分数做归一化，默认 True

	Returns:
		检索结果列表；执行失败时记录错误日志并返回 None
	"""
	try:
		# 创建权重融合排序器
		ranker = WeightedRanker(
			ranker_weights[0],
			ranker_weights[1],
			norm_score=norm_score
		)
		
		if output_fields is None:
			output_fields = ["text", "id"]
		
		search_result = milvus_client.hybrid_search(
			collection_name=collection_name,
			reqs=reqs,
			ranker=ranker,
			limit=limit,
			output_fields=output_fields,
		)
		
		total_hits = sum(len(hit) for hit in search_result) if search_result else 0
		logger.info(f"执行混合检索成功,总计处理{len(search_result) if search_result else 0}次查询")
		logger.info(f"\n 总计召回{total_hits}个查询结果")
		
		return search_result
	except Exception as e:
		logger.error(f"执行混合检索报错: {e}")
		return None


def query_chunks_by_chunk_id_list(
		milvus_client: MilvusClient,
		collection_name: str,
		chunk_id_list: List[str],
		output_fields: List[str]
):
	"""
	
	Args:
	    milvus_client:
		collection_name:
		chunk_id_list:
		output_fields:

	Returns:

	"""
	try:
		milvus_client.load_collection(collection_name)
		rows = milvus_client.query(
			collection_name=collection_name,
			output_fields=output_fields,
			ids=chunk_id_list,
		)
		return rows
	except Exception as e:
		logger.error(f"基于chunk id列表查询chunk异常: {e}")
		return []
	
