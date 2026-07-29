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
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config
from knowledge.processor.import_process.exception import ConfigurationError
from knowledge.processor.import_process.log_config import setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.utils.markdown_utils import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
	name = "document_split_node"
	
	def process(self, state: ImportGraphState) -> ImportGraphState:
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
		"""
		
		# 1. 从图状态中读取md_content
		config = get_import_config()
		md_content, md_file_name, file_title, max_content_length, min_content_length = self.get_node_info(state, config)
		
		# 2. 先根据MD标题H1-H6切分为若干个Section
		section_list, has_title = self.splitMarkDownByHeader(md_content, md_file_name)
		
		# 3. 处理全文无任意一个H1-H6标题的情况
		if not has_title:
			section_list = [
				{"title": "无标题", "body": md_content, "file_title": file_title, "parent_title": file_title}]
			self.logger.info("全文无标题，作为单个 chunk 处理")
		
		# 4. 对于过大或者过小的section进行二次切分或者合并
		section_list = self.split_and_merge_section(section_list, max_content_length, min_content_length)
		print(json.dumps(section_list, ensure_ascii=False, indent=4))
	
		# 5. 组装提交给Embedding模型的文本
		chunks = self.assemble_chunks(section_list)
		
		# 6. 更新状态state
		state["chunks"] = chunks
		self.log_step(step_name="STEP-7",message="文档切分完成，更新state的chunks属性成功")
		
		# 7. 备份数据
		self.backup_chunks(state,chunks)
		return state
	
	def backup_chunks(self,state:ImportGraphState,chunks:List[Dict[str,Any]]):
		output_file_dir = state.get("file_dir","")
		md_path = state.get("md_path","")
		
		if not output_file_dir:
			self.logger.debug("未设置文件输出目录")
			md_path_obj = Path(md_path)
			output_file_dir = md_path_obj.parent
		
		try:
			os.makedirs(output_file_dir,exist_ok=True)
			output_json_path = os.path.join(output_file_dir,"./chunks.json")
			with open(output_json_path,"w",encoding="utf-8") as f:
				json.dump(chunks,f,ensure_ascii=False,indent=4)
				self.log_step(step_name="STEP-7",message=f"文档切分完成，备份切分结果至{output_file_dir}")
		except Exception as e:
				self.logger.warning(f"文件备份失败: {e}")
		
	def get_node_info(self, state: ImportGraphState, config):
		self.log_step(step_name="STEP-1", message="获取节点信息")
		md_content = state.get("md_content", "")
		if md_content:
			md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
		file_title = state.get("file_title", "")
		md_path_obj = Path(state.get("md_path", ""))
		md_file_name = md_path_obj.stem
		max_content_length = config.max_content_length
		min_content_length = config.min_content_length
		
		if max_content_length <= min_content_length:
			raise ConfigurationError(node_name=self.name, message="文档块最大长度不能小于最小长度")
		
		return md_content, md_file_name, file_title, max_content_length, min_content_length
	
	def splitMarkDownByHeader(self, md_content, md_file_name: str):
		"""
		基于正则匹配Header后按照H1-H6切分
		
		特殊处理：
			1. 对于文档不是以Header标题开头（一上来就是正文）那么以文档名称作为section的title和parent_title
			2. 对于H1标题的section，其parent_title设置为当前文档的名称
			
		Args:
			md_content: MD内容
			md_file_name: 文档名称

		Returns:
			sections：List 文档片段列表
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
				
				# 一级标题 or 一上来就是文本 此时父标题为空 默认设为文档标题
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
	
	def split_and_merge_section(self, section_list, max_content_length, min_content_length):
		"""
		二次切分场景（某一个section的body内容超出最大长度，需要给予LangChain进行二次切分）
		1. 文档块过大，防止语义被稀释,
		2. 需要自定义元数组
		
		二次合并场景（连续的若干个section的块都比较小，需要对来自同一标题下的section的body进行合并）
		1. 多个同源标题下小块可以合并
		2. 代词在上下文中连续出现，合并有利于精确定位代词
		3. 合并块不能超出模型的边界阈值
		4. 元数组和跨权限的数据不能合并
		
		Args:
			section_list: 基于H1-H6标题切分后返回的列表
			max_content_length: 配置最大长度
			min_content_length: 配置最小长度

		Returns:
			result: List

		"""
		self.log_step(step_name="STEP-3", message="二次切分和合并块")
		
		current_sections = []
		
		# 1. 切分大块
		for section in section_list:
			section_body = section.get("body","")
			# 先处理表格  防止大表格中tr td字符串被切分为不同section导致错乱
			section["body"] = MarkdownTableLinearizer.process(section_body)
			self.logger.info("检测当前section是否存在表格 进行处理")
			
			# 再进行大块切分
			text_spliter_result = self.split_long_section(section, max_content_length)
			current_sections.extend(text_spliter_result)
		self.logger.info(f"Langchain切分大块文档片段后总计{len(current_sections)}个块")
		
		# 2. 合并同源小块
		final_sections = self.merge_short_sections(current_sections, min_content_length)
		self.logger.info(f"合并同源小块文档后总计{len(final_sections)}个块")
		
		# 3. 返回切分合并处理后的结果
		return final_sections
	
	def split_long_section(self, section, max_content_length) -> List[Dict[str, Any]]:
		"""
		切分长的section块
		
		Args:
			section: Dict[str,str] 文档片段对象
			max_content_length: 片段最大长度

		Returns:
			sub_sections：切分后的section片段列表
		"""
		self.log_step(step_name="STEP-4", message=f"使用LangChain 文本切分器尝试切分大片段")
		
		section_title = section.get("title", "")
		section_body = section.get("body", "")
		section_parent_title = section.get("parent_title", "")
		section_file_title = section.get("file_title", "")
		
		# 获取Title加上Body的总长度 小于最大长度 直接返回不进行二次切分
		section_title = f"{section_title}\n\n"
		total_length = len(section_body) + len(section_title)  # 2800个字符
		if total_length <= max_content_length:
			return [section]
		
		# 二次切分 先计算最大可用chunk块的长度
		available_chunk_size = max_content_length - len(section_title)
		
		# Edge Case：如果section标题长度大于最大长度 这种块直接返回 因为我们处理的是body 不是title
		if available_chunk_size <= 0:
			return [section]
		
		# 基于langchain的text--spliter进行切分
		
		text_spliter = RecursiveCharacterTextSplitter(
			chunk_size=available_chunk_size,
			chunk_overlap=0,
			length_function=len,
			separators=["\n\n", "\n", " ", "。", "!", "?", "，", ",", ".", ""],
			keep_separator=False
		)
		
		chunks: List[str] = text_spliter.split_text(section_body)
		# 如果二次切分出来还是一整块 那么无需再次处理 直接返回原section
		# TODO：Langchain split_text源码 每一段都小于chunk_size 加起来也小于chunk_size
		if len(chunks) <= 1:
			return [section]
		
		# 如果二次切分后多块 那么二次构建section
		sub_sections = []
		for index, chunk in enumerate(chunks):
			sub_sections.append({
				"title": f"{section_title}-{index + 1}",
				"body": chunk,
				"parent_title": section_parent_title,
				"file_title": section_file_title,
				"part": f"{index + 1}"
			})
		
		return sub_sections
	
	def merge_short_sections(self, current_sections, min_content_length):
		"""
		贪心累加思想
		1. 判断当前section和遍历到的下一个section是否同源
			- parent title相同 说明是一个H标题下的子节点
			- parent title不相同 语义不一致 不进行合并
		2. 再次判断当前累加的section body长度是否小于配置的片段最小值
		
		满足上述1和2条件: 需要进行合并
		不满足1和2条件：
			1. 更新指针curr_section
			2. 说明这个section不需要处理，直接加入到结果数组中，
		Args:
			current_sections: langchain切分大文本后返回的所有片段列表（保证所有片段长度不会超出最大配置length）
			min_content_length: 最小值

		Returns:
			results：合并处理后的最终section列表

		"""
		self.log_step(step_name="STEP-5", message=f"尝试合并小片段chunks")
		if not current_sections:
			return current_sections
		
		# 设置初始指针
		current_section = current_sections[0]
		final_sections = []
		
		for next_section in current_sections[1:]:
			# 是否同源（多个H1标题源头一定相同 开头文本和第一个H1一定相同 因为它们的父标题都是文档标题）
			is_same_parent = current_section.get("parent_title") == next_section.get("parent_title")
			
			# 如果满足：1. 相同父标题表示来自同一父节点下面 2. 当前section的长度小于最小section长度
			if is_same_parent and len(current_section.get("body", "")) < min_content_length:
				# 当前section的body吃掉下一个section的body
				current_section["body"] = current_section["body"].rstrip() + "\n\n" + next_section["body"].lstrip()
			else:
				# 不满足条件则加入到结果数组中
				final_sections.append(current_section)
				# 更新指针 用于下一次遍历
				current_section = next_section
		
		"""
		解决最后一次for循环遍历完成后遗留current_section未添加到数组的问题
		12 34 45 7  遍历到next为7的时候 发现加不进去了将45合并 然后curr变为7 循环结束 加入7
		12 34 56 78 遍历到next为8的时候 发现可以加进去78合并 循环结束 加入78
		"""
		if current_section:
			final_sections.append(current_section)
		
		return final_sections
	
	def assemble_chunks(self, section_list):
		"""
		组装最终喂给向量模型的数据
		Args:
			section_list:

		Returns:

		"""
		self.log_step(step_name="STEP-6", message="组装数据准备提交给向量Embedding模型")
		chunks = []
		
		for section in section_list:
			section_title = section.get("title", "")
			section_body = section.get("body", "")
			section_parent_title = section.get("parent_title", "")
			section_file_title = section.get("file_title", "")
			section_part = section.get("part", "")
			
			chunk = {
				"content": f"{section_title}\n\n{section_body}",
				"parent_title": section_parent_title,
				"file_title": section_file_title,
			}
			
			if section_part:
				chunk["part"]=section_part
			
			chunks.append(chunk)
			
		return chunks


if __name__ == "__main__":
	setup_logging()
	node = DocumentSplitNode()
	with open("/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md", 'r', encoding="utf-8") as file:
		content = file.read()
	node.process({
		"file_title": "万用表RS-12的使用.md",
		"md_content": content,
		"md_path": "/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md"
	})
