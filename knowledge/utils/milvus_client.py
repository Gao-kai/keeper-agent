# 创建Milvus客户端
from typing import Optional

from pymilvus import MilvusClient
from logging import INFO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=INFO)
load_dotenv(override=True)



# # Milvus 连接地址
# MILVUS_URL = http: // localhost: 19530
# # 知识库切片集合名
# CHUNKS_COLLECTION = keeper_chunks
# # 实体名称集合名
# ENTITY_NAME_COLLECTION = keeper_graph_entity_names
# # 商品名称集合名
# ITEM_NAME_COLLECTION = keeper_item_names
# # 相似度度量方式
# MILVUS_METRIC_TYPE = COSINE
# # 最小余弦相似度阈值
# MILVUS_MIN_COSINE_SCORE = 0.75

milvus_client: Optional[MilvusClient]  = None
def get_milvus_client(uri: str):
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


