"""
商品名确认节点
"""
from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState


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
		original_query = state.get("original_query","")
		
		
		
	
	

class ItemNameExtractor:
	"""
	Why?
	1. 用户模糊、口语化输入统一处理
	2. 提示词组装后call LLM
	2. 商品名称提取 单个、多个
	3. 返回格式JSON校验
	4. 返回数据清洗 围栏
	"""
	def __init__():
		pass
	
	def extract_by_llm(self):
		pass
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	