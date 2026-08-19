"""
Milvus 混合检索

1. Milvus官方文档的各种搜索方式的实现思路及底层逻辑
2. ANN和KNN的区别
3. Milvus的混合搜索 需要结合BGE-M3的稀疏向量和稠密向量
4. 加权排序器的机制  https://milvus.io/docs/zh/v2.6.x/weighted-ranker.md#Weighted-Ranker
5. Milvus官方文档必须要认真阅读一遍
6. 基于BGE-M3模型生成的稀疏和稠密向量已经做了L2归一化
"""
from pymilvus import DataType

from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model, generate_hybrid_embeddings
from dotenv import load_dotenv
from knowledge.utils.milvus_client import get_milvus_client, create_hybrid_search_request, execute_hybrid_search
load_dotenv(override=True)

if __name__ == "__main__":
	
	# 定义集合
	collection_name = "hybrid_search_collection"
	documents = [
		"姚明是一名NBA篮球运动员",
		"火箭队退役了姚明的11号球衣",
		"小明很喜欢看NBA，也喜欢姚明"
	]
	
	# 调用BGE-M3嵌入模型生成稀疏向量和稠密向量
	bge_m3_model = get_bge_m3_embedding_model()
	hybrid_embeddings = generate_hybrid_embeddings(bge_m3_model, documents)
	dense = hybrid_embeddings["dense"]
	sparse = hybrid_embeddings["sparse"]
	
	# 获取Milvus客户端
	milvus_client = get_milvus_client()
	if milvus_client.has_collection(collection_name):
		milvus_client.drop_collection(collection_name)
	
	# 创建schema
	schema = milvus_client.create_schema(enable_dynamic_field=True)
	
	# 主键字段
	schema.add_field(
		field_name="id",
		datatype=DataType.INT64,
		is_primary=True,
		auto_id=True
	)
	
	schema.add_field(
		field_name="text",
		datatype=DataType.VARCHAR,
		max_length=1000
	)
	
	# 添加稠密向量字段
	schema.add_field(
		field_name="dense_vector",
		datatype=DataType.FLOAT_VECTOR,
		dim=len(dense[0])
	)
	
	# 添加稀疏向量字段
	schema.add_field(
		field_name="sparse_vector",
		datatype=DataType.SPARSE_FLOAT_VECTOR,
	)
	
	# 创建索引
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
	
	# 创建集合
	collection = milvus_client.create_collection(
		collection_name=collection_name,
		schema=schema,
		index_params=index_params
	)
	
	milvus_client.load_collection(collection_name=collection_name)
	
	data = []
	
	for index, doc in enumerate(documents):
		data.append({
			"text": doc,
			"dense_vector": dense[index],
			"sparse_vector": sparse[index]
		})
	
	result = milvus_client.insert(
		collection_name=collection_name,
		data=data
	)
	
	# 我的问题
	question = ["小明最喜欢的篮球运动员是谁？"]
	question_hybrid_embeddings = generate_hybrid_embeddings(bge_m3_model, question)
	question_dense = question_hybrid_embeddings["dense"][0]
	question_sparse = question_hybrid_embeddings["sparse"][0]
	
	reqs = create_hybrid_search_request(dense_vector=question_dense,
	                                    sparse_vector=question_sparse,
	                                    dense_params=None,
	                                    sparse_params=None,
	                                    expr=None,
	                                    limit=10)
	
	res = execute_hybrid_search(
		milvus_client=milvus_client,
		limit=5,
		reqs=reqs,
		collection_name=collection_name,
		output_fields=["text"],
		ranker_weights=[0.7, 0.9]
	)
