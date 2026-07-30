"""
商品名识别节点

通过提前将商品名有关chunk交给LLM生成精确的商品名称后，
调用BGE-M3模型生成稀疏向量和稠密向量存入Milvus向量数据库，
增加用户在询问和商品名相关问题时召回率和精确度。

非流式输出时千问模型必须强制设置enabled-thinks为false
LLM的缓存封装


"""
from knowledge.processor.import_process.base import BaseNode, T


class ItemNameRecognitionNode(BaseNode):
	def process(self, state: T) -> T:
		# 1. 参数校验

		# 2. 构建调用LLM大模型识别商品名上下文

		# 3. 调用LLM模型识别
		
		# 4. 调用向量模型生成商品名向量

		# 5. 存入Milvus向量数据库

		# 6. 更新state
		pass