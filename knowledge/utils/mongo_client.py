import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

mongo_client = None

def get_mongo_client(uri: str) -> MongoClient | None:
	"""获取 MongoDB 客户端（单例模式）。

	优先复用全局已创建的客户端实例，避免重复创建连接；
	若尚未创建，则使用传入的 uri 或环境变量 MONGO_URL 创建新的客户端。

	Args:
		uri (str): MongoDB 连接字符串，如 "mongodb://user:pass@localhost:27017/?authSource=admin"。
			若为空或 None，则回退到环境变量 MONGO_URL。

	Returns:
		MongoClient | None: 创建成功的 MongoClient 实例；
			若创建失败（如连接串非法、网络异常等），返回 None。

	Raises:
		无：内部已捕获所有异常并记录日志，不会向上抛出。

	Example:
		client = get_mongo_client("mongodb://booker:Manulife@localhost:27017/?authSource=admin")
		if client:
			db = client["mydb"]
	"""
	try:
		global mongo_client
		
		if mongo_client is not None and isinstance(mongo_client, MongoClient):
			return mongo_client
		
		mongo_server_uri = uri or os.getenv("MONGO_URL")
		
		mongo_client = MongoClient(mongo_server_uri)
		
		return mongo_client
	except Exception as e:
		logger.error(f"创建Mongo DB客户端报错: {e}")
		return None


if __name__ == "__main__":
	client:MongoClient = get_mongo_client(uri=os.getenv("MONGO_URL"))
	db = client['test']
	col = db["test_col"]
	col.insert_one({
		"name":"lilei",
		"age":18
	})