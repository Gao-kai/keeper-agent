"""
商品名确认节点
"""
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState


class ConfirmItemNameNode(BaseNode):

	name = "confirm_item_name_node"
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 获取原始问题从state中
		# LLM提取 对齐下游基于商品名各路检索 原始问题质量差 口语化 代词 错别字 表达不清楚
	
	

class ItemNameExtractor:
	pass
	
	# 1. 获取LLM客户端 构建提示词模版LangChain？