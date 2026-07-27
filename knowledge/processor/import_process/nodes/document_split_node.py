"""
工业级RAG核心架构

# 为什么需要文档切分？
1. 向量模型会将输入文本映射为一个固定维度的稠密向量，如果将整个文档都给Embedding模型处映射为一个单一向量，那么文档中细节会被稀释，通过切分可以将庞大的文档一个个切分为小的片段，这种单一高浓度的快更加容易在向量空间中被精确命中
2. 一次性将全部文档放入向量模型，虽然当前大模型都支持128K以上的上下文，但是会导致首字生成时间大幅度上升，产生严重交互延迟。
3. 还会导致提示词模版过长每次向大模型交互的时候都携带大量的上下文，增加token消耗，增加经济成本。切分后每次只会返回Top K个更加精确的小片段，大大减少token消耗。
4. LLM处理上下文时会产生迷失在中间的问题，对首尾的注意力高中间的低。如果检索出来的片段过长，会给大模型产生信息干扰，产生幻觉。
5. 切分的时候可以为每一个chunk都打上详细的元数据信息，不仅有利于构建知识图谱，也可以让系统在生成答案时附带准确的文件溯源引用，尤其适合对生成答案高精确度场景的审计要求
6. 单纯向量检索擅长模糊语义的检索，但是对于专有名词、特定代码片段的时候效果不佳。切分之后一边将向量存储在向量数据库提供精确召回，一部分被提取关键字存入知识图谱或者ES中提供精确召回。因此细粒度的切分是混合检索和重排序的必备步骤。


"""
import json
import re
from pathlib import Path
from typing import List, Dict

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config
from knowledge.processor.import_process.log_config import setup_logging
from knowledge.processor.import_process.state import ImportGraphState


class DocumentSplitNode(BaseNode):
	name = "document_split_node"
	def process(self, state: T) -> T:
		"""
		文档切分节点
		一切目的减少LLM的幻觉，提高检索命中率
		1. 切分后文档块嵌入语义模型更加准确
		2. 注入元数据，方便溯源
		3. 多路召回检索基础
		4. 减少发送给LLM的上下文，减少token成本
		
		Args:
			state: 图状态

		Returns:
			state: 图状态
		
		1. 先内容 后标题 都是文件名
		2. 一上来就是标题
		"""
		
		# 1. 从图状态中读取md_content
		config = get_import_config()
		md_content, file_title, max_content_length, md_file_name = self.get_node_info(state, config)
		
		# 1. 根据MD标题切分
		result = self.splitMarkDownByHeader(md_content, md_file_name)
		print(json.dumps(result, ensure_ascii=False, indent=4))
	
	# 2. section块太长 基于langchain实现二次切分
	# 2. section块太短 尽可能合并同源信息
	
	# 3. 组装提交给Embedding模型的文本
	
	# 4. 更新状态state
	def get_node_info(self, state: ImportGraphState, config):
		self.log_step(step_name="STEP-1", message="获取节点信息")
		md_content = state.get("md_content", "")
		if md_content:
			md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
		file_title = state.get("file_title", "")
		max_content_length = config.max_content_length
		md_path_obj = Path(state.get("md_path", ""))
		md_file_name = md_path_obj.stem
		
		return md_content, file_title, max_content_length, md_file_name
	
	def splitMarkDownByHeader(self, md_content, md_file_name:str):
		"""
		基于正则匹配Header后按照H1-H6切分
		Args:
			md_content: MD内容
			md_file_name: 文档名称

		Returns:
		
		"""
		self.log_step(step_name="STEP-2", message="基于正则匹配截取Markdown")
		md_lines: List[str] = md_content.split("\n")
		
		is_code_fence = False
		# 正则会捕获两个组
		# 第一个组捕获到 一个到多个#
		# 第二个组捕获到 标题内容
		heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
		current_level = 0
		has_title = False
		current_title = ""
		section_list: List[Dict[str, str]] = []
		prev_body_lines = []
		# 存放当前已经遍历过的Header列表 索引0不使用 从1-6进行匹配
		heading_record = [""] * 7
		
		def savePrevHeadingSection():
			prev_body_text = "\n".join(prev_body_lines)
			# 将多个连续空行替换为单个空行
			prev_body_text = re.sub(r'\n\s*\n', '\n\n', prev_body_text)
			
			if prev_body_text or current_title:
				parent_title = ""
				for idx in range(current_level - 1, 0, -1):
					parent_title = heading_record[idx]
					if parent_title:
						break
				
				if not parent_title:
					parent_title = md_file_name
				
				section_list.append({
					"title": current_title or md_file_name,
					"body": prev_body_text,
					"parent_title": parent_title,
					"file_title": md_file_name
				})
		
		for md_line in md_lines:
			# 1. 如果是代码块中的H1-H6标题 不做切分处理
			if md_line.startswith("```") or md_line.startswith("~~~"):
				is_code_fence = not is_code_fence
				"""
				1 ```python
				2 print("hello world!")
				3 ```
				4 ## 总结
				
				解析：
				遍历到第1行，发现是代码块开始，此时is_code_fence取反后变为True
				遍历到第2行，is_code_fence没变还是True
				遍历到第3行，发现是代码块闭合，此时is_code_fence取反后变为False
				遍历到第4行，此时is_code_fence值是false，才可以将H2加入到结果数组中（⚠️）
				"""
			
			# 2. 如果不是代码块中的H1-H6标题 并且正则匹配到这是一个H1-H6标题 做切分处理
			matched = heading_re.match(md_line) if not is_code_fence else None
			
			if matched:
				has_title = True
				# 2.1 先保存上一个header及其包裹的内容 后更新当前信息
				savePrevHeadingSection()
				
				# 2.2 获取当前header的标题全名
				current_title = md_line.strip()
				
				# 2.3 获取当前header的等级 2-6
				current_level = len(matched.group(1))
				
				# 2.4 更新 heading_record列表
				heading_record[current_level] = current_title
				
				# 2.5 需要 heading_record列表中所有大于当前level的值全部清空
				
				# ## H2标题
				# ### H3标题
				# #### H4标题A
				# ## H2标题
				# #### H4标题B(⚠️它的父标题应该是最近的H2而不是上面的H3标题)
				
				#   遇到H2 ["","","H2标题","","","",""] 清空大于H2的所有标题
				#   遇到H3 ["","","H2标题","H3标题","","",""] 清空大于H3的所有标题
				#   遇到H4 ["","","H2标题","H3标题","H4标题","",""] 清空大于H4的所有标题
				#   遇到H2 ["","","H2标题","H3标题","H4标题","",""] 清空大于H2的所有标题 -> ["","","H2标题","","","",""]
				# 🚩遇到H4 ["","","H2标题","","H4标题","",""] 如果上一步不清空 这里取父标题就会取到H2之前的H3 而不是H2
				for index in range(current_level + 1, 7):
					heading_record[index] = ""
				
				# 2.6 清空prev_body_lines
				prev_body_lines = []
			else:
				prev_body_lines.append(md_line)
		
		# 循环完成之后最后一个Header包裹的内容也需要Save起来
		savePrevHeadingSection()
		
		return section_list, has_title


if __name__ == "__main__":
	setup_logging()
	node = DocumentSplitNode()
	with open("/Users/artest/Desktop/shopkeeper/knowledge/test/test.md", 'r', encoding="utf-8") as file:
		content = file.read()
	node.process({
		"file_title": "test.md",
		"md_content": content
	})
