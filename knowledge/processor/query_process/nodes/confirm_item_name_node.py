"""
商品名确认节点
"""
import json
import re
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.log_config import setup_logging


class ConfirmItemNameNode(BaseNode):
	name = "confirm_item_name_node"
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		"""
		# 1. 获取用户原始输入
		# 2. 调用LLM提取商品名称
		# 3. 构建confirmed\options\No的不同分支
		# 4. 向量检索item_names
		Args:
			state:

		Returns:

		"""
		# 1. 提取用户输入
		original_query = state.get("original_query", "")
		
		# 2. LLM提取item_names
		item_name_extractor = ItemNameExtractor()
		extract_result = item_name_extractor.extract_by_llm(original_query, [])
		item_names = extract_result.get("item_names")
		rewritten_query = extract_result.get("rewritten_query")
		
		# 3. 基于查询到的item_names分别去向量数据库查询
		self.vector_search(item_names)
		
		
		return state
	
	def vector_search(self, item_names:List[str]):
		pass


import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ItemNameExtractor:
	"""
	Why?
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
			{
				"history_text": history_context or "暂无历史上下文",
				"query": original_query
			}
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
				"original_query": "huawei Mate Station S12和H3C LA2608的区别是什么？"
			}
		)
