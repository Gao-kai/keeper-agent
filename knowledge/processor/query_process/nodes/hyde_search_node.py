"""
Hypothetical Document Embedding
HYDE 假设性文档嵌入技术
让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升召回质量

LLM 每一个文档块生成N个问题后入库 用户问题检索最匹配的问题 找到chunk
LLM 每一个问题先生成一个假的答案 假的答案向量检索最匹配的块 找到chunk
"""
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.config import get_query_config
from knowledge.processor.query_process.exception import ValidationError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_PROMPT_TEMPLATE
from knowledge.utils.bge_m3_embedding_model import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.log_config import setup_logging
from knowledge.utils.milvus_client import get_milvus_client, create_hybrid_search_request, execute_hybrid_search


class HydeDocumentEmbeddingSearchNode(BaseNode):
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 1. 参数校验
		self.log_step("step_1", f"校验节点输入")
		config = get_query_config()
		item_names, rewritten_query = self.validate_inputs(state)
	
		
		# 2. 获取LLM客户端
		llm_client = get_llm_client()
		if llm_client is None:
			return state
		self.log_step("step_2", f"构建生成假设性问题的提示词模版")
		
		# 3. 构建生成假设性答案的提示词模版
		human_message = HYDE_PROMPT_TEMPLATE.format(
			query=rewritten_query,
			item_names=",".join(item_names)
		)
		
		system_message = f"假设你是一个{",".join(item_names)}方面的专家，请你用流畅清晰专业的方式的语言回答"
		
		self.log_step("step_3", f"调用LLM生成假设性问题的答案")
		response = llm_client.invoke(
			[
				SystemMessage(content=system_message),
				HumanMessage(content=human_message)
			]
		)
		
		if not response:
			return state
		
		
		# 4. 生成假设性答案
		hypothetical_answer = response.content
		self.log_step("step_4", f"LLM生成假设性问题的答案成功:\n\r {hypothetical_answer}")
		
		# 5. 将用户原始问题 + 假设性答案 联合后 生成混合向量
		self.log_step("step_5", f"LLM生成假设性问题转化为混合向量")
		bge_m3_model = get_bge_m3_embedding_model()
		if bge_m3_model is None:
			return state
		
		
		final_content = f"{rewritten_query}\n{hypothetical_answer}"
		
		rewritten_query_hybrid_embedding = generate_hybrid_embeddings(
			embedding_docs=[final_content],
			embedding_model=bge_m3_model
		)
		
		if rewritten_query_hybrid_embedding is None:
			return state
		
		# 6. 构建混合请求对象
		self.log_step("step_6", f"构建混合请求对象")
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
			limit=3
		)
		
		# 7. 执行混合检索
		self.log_step("step_7", f"执行混合检索")
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
		
		# 8. 更新state对象
		hyde_embedding_chunks = search_result[0] if search_result else []
		
		return {
			"hyde_embedding_chunks":hyde_embedding_chunks
		}
	
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
	print("开始测试假设性答案查询节点")
	state = {
		"rewritten_query": "H3C LA2608 室内无线网关怎么创建 WLAN-ESS 接口呢？",
		"item_names": ["H3C LA2608 室内无线网关"]
	}
	hydeSearchNode = HydeDocumentEmbeddingSearchNode()
	state = hydeSearchNode.process(state)
	print(state)

