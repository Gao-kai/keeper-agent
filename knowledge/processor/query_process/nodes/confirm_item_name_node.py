"""
商品名确认节点
"""
import json
import re
from typing import List, Dict, Any, Tuple
import os
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_m3_embedding_model import bge_m3_ef, get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.log_config import setup_logging
from knowledge.utils.milvus_client import get_milvus_client, create_hybrid_search_request, execute_hybrid_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_OPTION_SIZE = 3
HIGH_RELATED_SCORE = 0.7
MIDDLE_RELATED_SCORE = 0.6
MAX_SCORE_GAP = 0.15


class ConfirmItemNameNode(BaseNode):
	name = "confirm_item_name_node"
	
	def __init__(self):
		self.item_name_extractor = ItemNameExtractor()
		super().__init__()
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		
		"""
		如何提高RAG检索准确度？
		核心是在源头进行过滤，而不是在检索过滤上花功夫
		
		# 1. 获取用户原始输入
		# 2. 调用LLM提取商品名称
		# 3. 构建confirmed/options/No的不同分支
		# 4. 向量检索item_names
		Args:
			state:

		Returns:
		"""
		# 1. 提取用户输入
		original_query = state.get("original_query", "")
		
		# 2. LLM提取item_names
		extract_result = self.item_name_extractor.extract_by_llm(original_query, [])
		extracted_item_names = extract_result.get("item_names")
		rewritten_query = extract_result.get("rewritten_query")
		
		# 2.1 如果LLM提取到了商品名称
		if extracted_item_names:
			
			# 3. 基于查询到的item_names分别去向量数据库查询
			search_result = self.vector_search(extracted_item_names)
			
			# 4. 基于RAG压测评估得出的相似度距离评分过滤出交付给下游最可能相关的商品名或供用户选择商品名
			confirmed_item_names, optional_item_names, confirmed_items = self.filter_item_name_by_score_limit(
				search_result)
			
			# 5. 即使已经加入到confirmed_item_names但是将得分差异太大的过滤掉
			confirmed_item_names = self.filter_item_name_by_score_gap(confirmed_items)
		
		# 2.2 如果LLM没有提取到商品名称
		else:
			confirmed_item_names = []
			optional_item_names = []
		
		# 5， 决定state是继续还是结束
		self.decide(state, extracted_item_names, confirmed_item_names, optional_item_names, rewritten_query)
		
		return state
	
	@staticmethod
	def vector_search(extracted_item_names: List[str]):
		"""
		基于LLM提取出的商品名称列表进行Milvus的混合检索
		Args:
			extracted_item_names:

		Returns:

		"""
		search_result = []
		
		# 1. 获取Milvus客户端
		milvus_client = get_milvus_client()
		if milvus_client is None:
			return search_result
		
		# 2. 使用BGE-M3模型生成item_names的稀疏向量和稠密向量
		bge_m3_model = get_bge_m3_embedding_model()
		hybrid_embeddings_result = generate_hybrid_embeddings(
			embedding_model=bge_m3_model,
			embedding_docs=extracted_item_names
		)
		# 3. 加载集合
		item_names_collection_name = os.getenv("ITEM_NAME_COLLECTION")
		milvus_client.load_collection(collection_name=item_names_collection_name)
		
		"""
		hybrid_embeddings_result = {
			dense: [[第一个item_name],[第二个item_name]],
			sparse: [[第一个item_name],[第二个item_name]]
		}
		"""
		
		# 4. 基于Milvus客户端进行混合检索
		for index, extracted_item_name in enumerate(extracted_item_names):
			# 4.1 构建混合检索请求
			hybrid_search_requests = create_hybrid_search_request(
				dense_vector=hybrid_embeddings_result["dense"][index],
				sparse_vector=hybrid_embeddings_result["sparse"][index],
				dense_req_field_name="item_name_dense_vector",
				sparse_req_field_name="item_name_sparse_vector",
				limit=5
			)
			
			# 4.2 执行混合检索
			"""
			问题列表：
			1. 构建混合检索请求时的Limit和最终检索时候的limit不一样会怎么样
			2. BGE-M3嵌入模型默认只会对稠密向量进行L2归一化 arctan是什么
			3. 为什么稀疏向量无法进行归一化
			4. IP和COSINE是度量标准，为什么归一化后选哪个都一样
			5. 权重融合排序器weightedRanker的norm_score归一化是什么
			6. 混合检索的工作原理
			"""
			hybrid_search_result = execute_hybrid_search(
				milvus_client=milvus_client,
				limit=5,
				reqs=hybrid_search_requests,
				collection_name=item_names_collection_name,
				output_fields=["item_name"],
				ranker_weights=[0.7, 0.3],
				norm_score=True
			)
			
			print(hybrid_search_result)
			
			# 4.3 构建返回数据
			"""
			search_results:
			[
			    {
			        "extracted_item_name": "DT-9205A",
			        "matched_items": [
			            {"item_name": "DT-9205A", "score": 0.95},
			            {"item_name": "DT-9205B", "score": 0.72},
			            {"item_name": "DT-9202",  "score": 0.61},
			        ]
			    },
			    {
			        "extracted_item_name": "万用表",
			        "matched_items": [
			            {"item_name": "万用表UT61E", "score": 0.78},
			            {"item_name": "DT-9205A",    "score": 0.62},
			        ]
			    },
			]
			"""
			hits = hybrid_search_result[0]
			search_result.append({
				"extracted_item_name": extracted_item_name,
				"matched_items": [
					{"item_name": hit.get("entity", {}).get("item_name"), "score": hit.get("distance")} for hit in hits
				]
			})
		
		return search_result
	
	@staticmethod
	def decide(
			state: QueryGraphState,
			item_names: List[str],
			confirmed_item_names: List[str],
			optional_item_names: List[str],
			rewritten_query: str
	):
		"""
		
		Args:
			state:
			item_names:
			confirmed_item_names:
			optional_item_names:
			rewritten_query:

		Returns:

		"""
		# 确认成功：设置 item_names 和 rewritten_query
		# 不返回Answer给用户 走四路召回
		if confirmed_item_names and isinstance(confirmed_item_names, List):
			state["item_names"] = confirmed_item_names
			state["rewritten_query"] = rewritten_query
		elif optional_item_names and isinstance(optional_item_names, List):
			# 有候选：设置 answer 为反问文案
			# 返回Answer给用户 走直接返回
			state["answer"] = f"""
			我不能确定您所说的具体是哪一款产品。
			您指的是不是下面这些产品?
			{"\n".join(optional_item_names)}
			如果是请告诉我具体的产品名称！
			"""
		else:
			# 直接返回 有Answer
			state['answer'] = "抱歉，我无法识别您咨询的具体产品名称，请提供更加准确的产品名称或型号。"
	
	@staticmethod
	def filter_item_name_by_score_limit(search_result: List[Dict[str, Any]]) -> Tuple[
		List[str], List[str], List[Dict[str, Any]]]:
		"""
		基于向量数据库检索到的商品名称，配合评分阈值来将商品名放入对应的confirmed或options数组中
		
		阈值怎么来？（RAG评估的框架）
			1. 先构建100个用户最常见的问题构成测试数据集
			2. 构建50对confirmed和optional的阈值对[0.75.0.63]类似
			3. 挑选出准确率最高的阈值对作为后续评分的阈值
			
		分数阈值确定之后的作用：
			1. 如果matched数组中某个item_name的得分大于0.7 代表可以进入confirmed数组
			2. 如果matched数组中某个item_name的得分大于0.6 代表可以进入options数组
			
		Args:
			search_result:

		Returns:
			confirmed有值 options无值 表示找到了用户问题中确定的商品名称 传递给下游
			confirmed无值 options有值 询问用户
			confirmed和options都没有值 告诉用户重新询问 没有找到
			confirmed和options都有值 也以confirmed为准
		
		去重逻辑：
			confirmed已经有的商品名称，不能再添加相同名称进去
			options已经有的商品名称，不能再添加同名进去
			商品名称为A已经加入到了confirmed中，后续这个商品名称A既不能加入到confirmed中也不能加入到options中
			商品名称为A已经加入到了options中，后续这个商品名称A不能加入到options中，但是可以加入到confirmed中
		
			[
			    {
			        "'extracted_item_name'": "DT-9205A",
			        "matched_items": [
			            {"item_name": "DT-9205A", "score": 0.95},
			            {"item_name": "DT-9205B", "score": 0.72},
			            {"item_name": "DT-9202",  "score": 0.61},
			        ]
			    },
			    {
			        "'extracted_item_name'": "万用表",
			        "matched_items": [
			            {"item_name": "万用表UT61E", "score": 0.78},
			            {"item_name": "DT-9205A",    "score": 0.62},
			        ]
			    },
			]
		"""
		confirmed_item_names = []
		optional_item_names = []
		confirmed_items = []  # 为了下一步得分差异过大的过滤
		
		for index, item in enumerate(search_result):
			# 获取从Milvus向量数据库检索到的和LLM识别出的商品名最匹配的matches列表
			extracted_name = item.get("extracted_item_name")
			matches = item.get("matched_items", [])
			if not matches:
				continue
			
			# 按照score倒序排列
			sorted_by_score_matches = sorted(matches, key=lambda m: m["score"], reverse=True)
			
			# 进行阈值过滤
			high_related_item_names = [match for match in sorted_by_score_matches if
			                           match.get("score", 0) >= HIGH_RELATED_SCORE]
			
			if high_related_item_names:
				# 查找在milvus检索出来的高相似的列表中是否存在和LLM提取出的extracted_name完全匹配的那条Item
				accurate_item = None
				for high_related_item_name in high_related_item_names:
					item_name = high_related_item_name.get("item_name")
					if item_name == extracted_name:
						accurate_item = item_name
						break
				
				# 场景1: 找到精确匹配的商品名
				if accurate_item:
					picked = accurate_item["item_name"]
					if picked not in confirmed_item_names:
						confirmed_item_names.append(picked)
						confirmed_items.append(accurate_item)
				elif len(high_related_item_names) == 1:
					# 场景2: 得分大于confirm阈值的商品只有一个 那就认为它是准确的
					picked = high_related_item_names[0]['item_name']
					if picked not in confirmed_item_names:
						confirmed_item_names.append(picked)
						confirmed_items.append(high_related_item_names[0])
				else:
					# 场景3:  得分大于confirm阈值的商品有多个 那么选出前Max个加入到options中 让用户选择
					for high_related_item_name in high_related_item_names[:MAX_OPTION_SIZE]:
						picked = high_related_item_name.get("item_name")
						if picked not in optional_item_names and picked not in confirmed_item_names:
							optional_item_names.append(picked)
			else:
				# 除开高相似的item 剩余的都是小于HIGH_RELATED_SCORE的item
				# 过滤出MIDDLE_RELATED_SCORE也就是可以加入到options中的item
				mid_related_item_names = [match for match in matches if match.get("score") >= MIDDLE_RELATED_SCORE]
				
				# 场景4: 得分大于options阈值的商品中取出MAX_OPTION_SIZE个 加入到用户选择列表中
				if mid_related_item_names:
					for mid_related_item_name in mid_related_item_names[:MAX_OPTION_SIZE]:
						picked = mid_related_item_name.get("item_name")
						if picked not in optional_item_names and picked not in confirmed_item_names:
							optional_item_names.append(picked)
		
		return confirmed_item_names, optional_item_names[:MAX_OPTION_SIZE], confirmed_items
	
	@staticmethod
	def filter_item_name_by_score_gap(confirmed_items: List[Dict[str, Any]]) -> List[str]:
		"""
		假设有多个item刚好都进入到confirmed_item_names数组中
		但是可能是下面这种情况：
		[
			{"item_name": "DT-9205A", "score": 0.95},
            {"item_name": "DT-9205B", "score": 0.92},
            {"item_name": "DT-9202",  "score": 0.71},
		]
		
		可以看出最后一个虽然也大于0.7阈值，但是相对于前面两个其得分的差距过大
		因此一旦超出最大得分的差距MAX_SCORE_GAP
		还需要将这种item进行过滤
		
		Args:
			confirmed_items:


		Returns:

		"""
		
		if len(confirmed_items) == 0:
			return []
		
		# 找到最大值
		max_score = 0
		for confirmed_item in confirmed_items:
			score = confirmed_item.get("score")
			if score > max_score:
				max_score = score
		
		# 将超出最大值差异阈值的直接过滤掉
		return [confirmed_item.get("item_name") for confirmed_item in confirmed_items if
		        max_score - confirmed_item.get("score") <= MAX_SCORE_GAP]


class ItemNameAligner:
	"""
	1. 查询向量数据库
	2. 评分对齐
	3. 差异过滤
	"""


class ItemNameExtractor:
	"""
	1. 用户模糊、口语化输入统一处理
	2. 提示词组装后call LLM
	2. 商品名称提取 单个、多个
	3. 返回格式JSON校验
	4. 返回数据清洗 围栏
	"""
	
	def extract_by_llm(self, original_query: str, history_context: List[Dict]) -> Dict[str, Any]:
		"""
		基于LLM从用户输入中提取商品名称
		Returns:

		"""
		# 初始化兜底返回值
		result = {
			"item_names": [],
			"rewritten_query": original_query
		}
		
		# 获取LLM客户端
		llm_client = get_llm_client(response_json=True)
		if llm_client is None:
			return result
		
		# 构建提示词模版
		system_message = "你是一个专业的客服助手，擅长从用户问题中做出准确的意图识别和关键信息提取"
		human_message = ITEM_NAME_EXTRACT_TEMPLATE.format(
			history_text=history_context or "暂无历史上下文",
			query=original_query
		)
		
		# LLM调用提取商品名称
		try:
			response = llm_client.invoke([
				SystemMessage(content=system_message),
				HumanMessage(content=human_message)
			])
			
			if not response.content:
				return result
			
			# 清洗LLM返回结果 保证数据结构合法
			parsed_response = self.parse_llm_response(response.content)
			result["item_names"] = parsed_response.get("item_names")
			result["rewritten_query"] = parsed_response.get("rewritten_query")
		except Exception as e:
			logger.error(f"LLM 提取商品名结果异常:{e}")
		
		return result
	
	@staticmethod
	def parse_llm_response(response_content: str) -> Dict[str, Any]:
		
		# 清洗Json围栏
		parsed_content = re.sub(r"^```(?:json)?\s*", "", response_content.strip())
		parsed_content = re.sub(r"\s*```$", "", parsed_content)
		
		# json反序列化
		try:
			parsed_result: Dict[str, Any] = json.loads(parsed_content)
		except Exception as e:
			raise ValueError(f"JSON反序列化失败:{e}")
		
		# 对JSON对象字段再次清洗防止出现异常数据
		raw_item_names = parsed_result.get("item_names")
		safe_item_names = [str(item_name).strip() for item_name in raw_item_names] if isinstance(raw_item_names,
		                                                                                         list) else []
		raw_rewritten_query = parsed_result.get("rewritten_query")
		safe_rewritten_query = str(raw_rewritten_query).strip() if isinstance(raw_rewritten_query, str) else ""
		
		# 返回结果
		return {
			"item_names": safe_item_names,
			"rewritten_query": safe_rewritten_query
		}


if __name__ == "__main__":
	setup_logging()
	node = ConfirmItemNameNode()
	node.process(
		{
			# "original_query": "huawei Mate Station S12和H3C LA2608的区别是什么？",
			"original_query": "你们店里面的Iphone17现在多少钱呢？"
		}
	)
