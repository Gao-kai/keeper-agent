"""
混合向量检索节点
"""


from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.config import get_query_config
from knowledge.processor.query_process.exception import ValidationError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.log_config import setup_logging
from knowledge.utils.milvus_client import get_milvus_client, create_hybrid_search_request, execute_hybrid_search


class HybridVectorSearchNode(BaseNode):
	name = "hybrid_vector_search_node"
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 1. 参数校验
		config = get_query_config()
		item_names, rewritten_query = self.validate_inputs(state)
		self.log_step("step_1", f"参数校验")
		
		# 2. 将上一节点大模型重写后的用户问题进行向量化
		bge_m3_model = get_bge_m3_embedding_model()
		if bge_m3_model is None:
			return state
		
		rewritten_query_hybrid_embedding = generate_hybrid_embeddings(
			embedding_docs=[rewritten_query],
			embedding_model=bge_m3_model
		)
		self.log_step("step_2", f"用户问题成功转化为混合向量")
		
		if rewritten_query_hybrid_embedding is None:
			return state
	
		# 3. 构建混合查询请求
		# 可以在搜索请求中包含过滤条件，以便 Milvus 在进行 ANN 搜索前进行元数据过滤，将搜索范围从整个 Collections 缩小到只搜索符合指定过滤条件的实体
		milvus_client = get_milvus_client()
		if milvus_client is None:
			return state
		
		expr = "item_name IN {item_names}"
		expr_params = {"item_names": item_names}
		reqs = create_hybrid_search_request(
			dense_vector=rewritten_query_hybrid_embedding["dense"][0],
			sparse_vector=rewritten_query_hybrid_embedding["sparse"][0],
			dense_req_field_name="dense_vector",
			sparse_req_field_name="sparse_vector",
			dense_params=None,
			sparse_params=None,
			expr=expr,
			expr_params=expr_params,
			limit=5
		)
		
		
		# 4. 执行混合检索
		chunk_collection_name = config.chunks_collection
		milvus_client.load_collection(collection_name=chunk_collection_name)
		search_result = execute_hybrid_search(
			milvus_client=milvus_client,
			limit=5,
			reqs=reqs,
			collection_name=chunk_collection_name,
			output_fields=["chunk_id", "content", "item_name"],
			ranker_weights=[0.5, 0.5],
			norm_score=True
		)
		
		if search_result is None:
			return state
		
		# 5. 更新state
		chunks = search_result[0] if search_result else []
		self.log_step("step_3", f"搜索完成，返回 {len(chunks)} 条结果")
		
		return  {"embedding_chunks": chunks}
	
	def validate_inputs(self, state: QueryGraphState):
		item_names = state.get("item_names")
		rewritten_query = state.get("rewritten_query")
		
		if not item_names or not isinstance(item_names, list):
			raise ValidationError(node_name=self.name, message=f"输入参数 [item_names] 校验失败")
		
		if not rewritten_query or not isinstance(rewritten_query, str):
			raise ValidationError(node_name=self.name, message=f"输入参数 [rewritten_query] 校验失败")
		
		return item_names, rewritten_query


if __name__ == '__main__':
	setup_logging()
	print("开始测试查询节点-向量混合检索节点")
	state = {
		"rewritten_query":"H3C LA2608 室内无线网关怎么创建 WLAN-ESS 接口呢？",
		"item_names": ["H3C LA2608 室内无线网关"]
	}
	hybridVectorSearchNode = HybridVectorSearchNode()
	state = hybridVectorSearchNode.process(state)
	print(state)
	
