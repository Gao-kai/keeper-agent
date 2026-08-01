from pymilvus import MilvusClient, DataType

from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model

if __name__ == "__main__":
	# 创建Milvus客户端
	milvus_client = MilvusClient(
		uri="http://localhost:19530"
	)
	
	# 创建Schema
	schema = milvus_client.create_schema(enable_dynamic_field=True)
	
	# 添加主键字段
	schema.add_field(
		field_name="id",
		datatype=DataType.INT64,
		is_primary=True,
		auto_id=False,
	)

	# 添加向量字段
	schema.add_field(
		field_name="my_vector",
		datatype=DataType.FLOAT_VECTOR,
		dim=1024
	)

	# 添加标量字段
	schema.add_field(
		field_name="row_text",
		datatype=DataType.VARCHAR,
		max_length=2048
	)
	
	# 创建索引
	index_params = milvus_client.prepare_index_params()
	
	# 添加稠密向量索引字段
	index_params.add_index(
		field_name="my_vector",  # Name of the vector field to be indexed
		index_type="IVF_FLAT",  # Type of the index to create
		index_name="dense_vector_index",  # Name of the index to create
		metric_type="COSINE",  # Metric type used to measure similarity
		params={
			"nlist": 64,  # Number of clusters for the index
		}  # Index building params
	)
	
	# 创建集合Collection
	collection_name = "item_name_collection"
	
	if milvus_client.has_collection(collection_name=collection_name):
		milvus_client.drop_collection(collection_name=collection_name)
		
	collection = milvus_client.create_collection(
		collection_name=collection_name,
		schema=schema,
		index_params=index_params
	)
	
	
	# 准备数据
	docs = [
		"姚明是一名篮球远动员",
		"打篮球有助于身体健康",
		"NBA是世界上最成功的篮球职业联盟"
	]
	
	# 获取向量模型
	embedding_model = get_bge_m3_embedding_model()
	docs_embeddings = embedding_model.encode_documents(docs)
	
	data = []
	for index,doc in enumerate(docs):
		data.append({
			"id":index+1,
			"my_vector":docs_embeddings["dense"][index].tolist(),
			"row_text":doc[:2048]
		})
	

	# 插入数据
	insert_res = milvus_client.insert(
		collection_name=collection_name,
		data=data
	)
	
	print(f"插入成功:{insert_res}")
	
	# 插入数据后，强制刷新数据到磁盘
	milvus_client.flush(collection_name=collection_name)
	
	# 检索之前加载集合
	milvus_client.load_collection(collection_name=collection_name)
	
	# 检索数据
	query_embedding = embedding_model.encode_queries(["姚明是做什么的？"])
	search_result = milvus_client.search(
		collection_name=collection_name,
		data=query_embedding["dense"],
		limit=2,
		output_fields=["id","row_text"]
	)
	
	print(f"检索结果:{search_result}")
	
