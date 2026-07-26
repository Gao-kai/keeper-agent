"""
MD中图片链接处理

1. 都加上图片描述 VLM多模态去做这个事情
2. 向量切分入库的时候存
3. 查询的时候更加精确
4. 图片地址处理要放在Minio对象存储 查询的时候访问公网IP 回填过来

# 对象存储的优点
ak sk entry_point

# TODO 并行处理图片摘要 + 大图片本地压缩
"""
import base64
import os
import mimetypes
import re
import time
from collections import deque
from pathlib import Path
from pyexpat.errors import messages
from typing import Tuple, List, Deque, Dict
from minio import Minio

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.config import get_import_config, ImportConfig
from knowledge.processor.import_process.exception import ValidationError, FileProcessingError
from knowledge.processor.import_process.log_config import setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_client import get_minio_client


class MDImageNode(BaseNode):
	name = "md_image_node"
	
	def process(self, state: T) -> T:
		"""
		MD文档中Image图片处理
		Args:
			state: 上一节点更新后图状态

		Returns:
			state: 当前节点更新后的图状态
		"""
		
		# 1. 读取MD文件内容，处理文件路径
		md_content, md_path_obj, images_dir_obj = self._get_md_info(state)
		
		if not images_dir_obj.exists():
			self.logger.info(f"文件{md_path_obj.name}没有图片需要处理")
			state["md_content"] = md_content
			return state
		
		# 2. 扫描并处理图片(减少VLM大模型噪音)
		target_images_list = self._scan_and_filter_images(md_content, md_path_obj, images_dir_obj)
		if not target_images_list:
			self.logger.info(f"文件{md_path_obj.name}中未找到需要处理的有效图片引用")
			return state
		
		# 3. VLM模型生成图片描述
		images_summarise = self._generate_image_summaries(md_path_obj, target_images_list)
		print(images_summarise)
		# images_summarise = {
		#     '820246ff8d6f448489eb36a1297e028a8ca8ff17100aecb8aa38d685b069fc19.jpg': '直流电流测量接线示意图',
		#     'eb330b9b1b79716fc9feacd30808b4ed86c395ebaab24c29dec65d68060e10be.jpg': '直流电流测量接线示意图',
		#     '3dce15efe5689c2d8c904dfbbb653eef71ce00b270bd08d47e3b474af2fb68a8.jpg': '万用表电阻测量',
		#     '87ccf38edca64aa36a803d7091a6385340dace221ee3105f93037e3b4285d161.jpg': '最大电压限制标识',
		#     '84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg': '交流电压测量示意图',
		#     'f5c6db12e9569ee13bd78fd38747397dd535fbb72326e87b43deaa240e8ec70b.jpg': '双绝缘保护标识',
		#     'a6c9fcfb41cfc997c0f88e35c06f34818e7b53d83d92fa3580257172c1c24ec7.jpg': 'RSPro品牌标志',
		#     '2ec4f0e7e8b05c73503dc25db1ad0e65b99a06b2dea8d8a9525a1865c75095e0.jpg': '警告标识',
		#     '9cc6ec399a5591e7939d410fc4f8c64396dd6935ac92a4830442fd8a69ff71de.jpg': '危险电压标识',
		#     'f5470dcb3ab08b06212db41f4b4b4728bcd0d46a0bafcae45dbee4b840a4ec65.jpg': '二极管测试指示符号',
		#     '16f1ae918c8905b9e4fc1d0c08fe8a9b55c34d69f4c1191f790bf08571419299.jpg': '中文语言标识',
		#     'd6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg': '万用表RS-12外观及端口示意图',
		#     '17a896b47789994e2944e1590940a56fff9a93c68fed9b924cd572f4917cf087.jpg': 'RS-12数字万用表外观图'
		# }
		
		# 5. 将图片上传至Minio服务器
		minio_client = get_minio_client()
		remote_image_urls = self._upload_images_to_minio(target_images_list, minio_client, md_path_obj,
		                                                 config=get_import_config())
		
		# 6.  回写图片摘要和图片链接到md_content
		new_md_content = self._replace_summary_and_remote_url(remote_image_urls, images_summarise, md_path_obj,
		                                                      md_content)
		with open(md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}"), 'w', encoding='utf-8') as f:
			f.write(new_md_content)
		self.logger.info(f'MD文档替换完成')
		
		# 7. 更新state
		state['md_content'] = new_md_content
		return state
	
	def _get_md_info(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
		"""
		基于文件系统读取md文档内容，获取Path对象和md文档相关联的images存储Path对象
		Args:
			state: 图状态

		Returns:
			md_content: md文档内容
			md_path_obj: md路径对象
			images_path_obj: md相对应的图片保存路径对象

		"""
		md_path = state.get("md_path", "")
		
		if not md_path:
			raise ValidationError(node_name=self.name, message=f"{md_path}文档路径为空")
		
		md_path_obj = Path(md_path)
		
		if not md_path_obj.exists():
			raise FileProcessingError(node_name=self.name, message=f"{md_path}文档路径不存在")
		
		with open(md_path_obj, "r", encoding="utf-8") as f:
			md_content = f.read()
		
		images_dir_obj = md_path_obj.parent / "images"
		
		return md_content, md_path_obj, images_dir_obj
	
	def _scan_and_filter_images(self, md_content, md_path_obj, images_dir_obj) -> List[
		Tuple[str, str, Tuple[str, str, str]]]:
		"""
		扫描图片返回图片相关上下文信息

		1. 先找到当前图片所属的最近的标题
		2. 从图片内容的上一行开始向上找，找到最近的一级标题之间的是上文（可能标题不存在）
		3. 从图片内容的下一行开始向下找，找到最近的一级标题之间的是下文
		4. 基于段落和最大字符选择从这个区域留下多少字符

		Args:
			md_content:
			md_path_obj:
			images_dir_obj:

		Returns:
			image_name
			image_path
			image_context
		"""
		config = get_import_config()
		target_images_list = []
		
		for image_name in os.listdir(images_dir_obj):
			
			# 图片后缀校验
			image_ext = os.path.splitext(image_name)[1]
			if image_ext not in config.image_extensions:
				continue
			
			# 构建完整图片路径
			image_path = str(images_dir_obj / image_name)
			
			# 图片上下文提取(包含标题 上文 下文)
			image_context: List[Tuple[str, str, str]] = self._extract_image_context_with_limit(md_content, image_name,
			                                                                                   max_chars=config.max_image_context_length)
			if not image_context:
				self.logger.debug(f"图片{image_name}未在{md_path_obj.name}中引用")
				continue
			# 一张图片可能会被多次引用 只取第一个引用的上下文
			primary_image_context = image_context[0]
			
			# 图片信息保存（包含图片名称 图片路径 图片上下文）
			target_images_list.append((image_name, image_path, primary_image_context))
		
		self.logger.info(f"共有{len(target_images_list)}张图片在markdown中引用处理")
		return target_images_list
	
	def _extract_image_context_with_limit(self, md_content: str, image_name: str, max_chars: int = 100) -> List[
		Tuple[str, str, str]]:
		"""
		基于正则提取图片在MD上下文
		1. 首先基于图片名在整个md_content中查找这个图片的位置 基于正则
		![](images/a71e4d4b0726a0ea9a37fe4fc3793c47911ca2f8e3582cd4f33cc122f3392baf.jpg "图片hover提示")
		!\[.*?\]\(.*?{image_name}.*?\)
		2. 向上找离图片最近的标题，如果没有查询到开始（正则查标题）
		3. 向下找离图片最近的标题，如果没有查询到末尾（正则查标题）
		4. 截取图片到上下标题之前内容，交给下一个方法处理
		Returns:
		"""
		md_image_pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)")
		md_header_pattern = re.compile(r"^#{1,6}\s+")
		lines = md_content.split("\n")
		context_list = []
		for line_index, line_content in enumerate(lines):
			# 查找当前图片在markdown内容中是否存在
			if not md_image_pattern.search(line_content):
				continue
			
			# 首先找自下向上遍历图片到最近的第一个标题的内容和索引【锁定上文区间】
			prev_header_content = ""
			prev_header_index = -1
			for index in range(line_index - 1, -1, -1):
				if md_header_pattern.search(lines[index]):
					prev_header_content = lines[index].strip()
					prev_header_index = index
					break
			
			# 提取上文
			pre_start_index = prev_header_index + 1
			pre_end_index = line_index
			pre_lines = lines[pre_start_index:pre_end_index]
			pre_context = self._extract_context_with_limit(
				pre_lines,
				max_chars,
				direction="backward"
			)
			
			# 自上到下遍历图片到最近的第一个标题索引【锁定下文区间】
			next_header_index = len(lines)
			for index in range(line_index + 1, len(lines)):
				if md_header_pattern.search(lines[index]):
					next_header_index = index
					break
			
			# 提取下文
			next_start_index = line_index + 1
			next_end_index = next_header_index
			next_lines = lines[next_start_index:next_end_index]
			next_context = self._extract_context_with_limit(
				next_lines,
				max_chars,
				direction="forward"
			)
			
			context_list.append(
				(prev_header_content,
				 pre_context,
				 next_context)
			)
		return context_list
	
	def _extract_context_with_limit(self, extract_lines: List[str], max_chars: int, direction: str) -> str:
		"""
		按照最大字符限制提取上下文
		markdown中使用\n来区分段落，使用两个空格来区分行
		因此需要将多个连续的非空行合并为一个段落

		Args:
			extract_lines: 提取出来的图片上下文
			max_chars: 最大字符
			direction: 查询方向  "forward" 从前往后，"backward" 从后往前。

		Returns:
		"""
		final_context_list = []
		curr_context_list = []
		for line in extract_lines:
			stripped_line = line.strip()
			# 如果遇到某一行为空字符串 此时视为之前收集的是一个段落 将其转化为字符串放入final_context_list
			if stripped_line == "":
				if curr_context_list:
					final_context_list.append("\n".join(curr_context_list))
					curr_context_list = []
			else:
				# 如果遇到其他图片 也认为是一个分隔符
				if re.match(r"^!\[.*?]\(.*?\)$", stripped_line):
					if curr_context_list:
						final_context_list.append("\n".join(curr_context_list))
						curr_context_list = []
					# 含有图片标记的这一行不需要收集 因为他不是上下文 也不是有用信息
					continue
				
				# 如果既不是图片行 也不是空行的 才收集进来
				curr_context_list.append(stripped_line)
		
		# 全部遍历完成之后 有可能与遇到最后一段下面不是空行的
		if curr_context_list:
			final_context_list.append("\n".join(curr_context_list))
		
		if not final_context_list:
			return ""
		
		# "backward" 从后往前提取上文的时候 先翻转列表 保证最靠近图片的行最先被提取出来
		if direction == "backward":
			final_context_list.reverse()
		
		# 在字数限制范围内尽可能截图更多的行
		selected_line = []
		total = 0
		
		for context_line in final_context_list:
			context_line_len = len(context_line)
			if total + context_line_len > max_chars and selected_line:
				break
			selected_line.append(context_line)
			total += context_line_len
		
		if direction == "backward":
			selected_line.reverse()
		
		return "\n\n".join(selected_line)
	
	def _generate_image_summaries(self, md_path: Path, target_images_list: List[Tuple[str, str, Tuple[str, str, str]]]):
		"""
 
		Args:
			md_path: MD文档路径对象
			target_images_list: 待生成摘要的图片列表
 
		Returns:
			 Dict
		"""
		image_summaries = {}
		config = get_import_config()
		try:
			from openai import OpenAI
			vml_client = OpenAI(
				api_key=config.openai_api_key,
				base_url=config.openai_api_base
			)
		except ImportError:
			self.logger.error("未安装OPENAI库")
			return image_summaries
		except Exception as e:
			self.logger.info("初始化VLM客户端失败", e)
			return image_summaries
		
		request_timestamps: Deque[float] = deque()
		window_seconds = 60
		max_request = config.requests_per_minute or 10
		for item_image in target_images_list:
			image_name, image_path, primary_image_context = item_image
			
			self._enforce_rate_limit(request_timestamps, max_request, window_seconds)
			self.logger.info(f"正在生成摘要: {image_name}")
			summary_text = self._call_vlm_generate_summary(
				config,
				client=vml_client,
				image_name=image_name,
				image_path=image_path,
				image_context_tuple=primary_image_context,
				md_file_name=md_path.stem,
			
			)
			image_summaries[image_name] = summary_text
		
		return image_summaries
	
	def _call_vlm_generate_summary(self, config: ImportConfig, client, image_name, image_path, image_context_tuple,
	                               md_file_name) -> str:
		"""
		调用VLM大模型生成图片摘要
		Args:
			config:
			client:
			image_name:
			image_path:
			image_context_tuple:
			md_file_name:

		Returns:

		"""
		base64_image_data_url = self._image_to_base64_data_url(image_path)
		
		# 上下文解包
		prev_header_content, pre_context, next_context = image_context_tuple
		
		# 构建提示词中上下文片段
		context_list = []
		if prev_header_content:
			context_list.append(f"所属图片章节: {prev_header_content}")
		if pre_context:
			context_list.append(f"图片上文: {pre_context}")
		if next_context:
			context_list.append(f"图片下文: {next_context}")
		
		context_info = "\n".join(context_list) if context_list else "该图片无可用上下文"
		
		# 调用VLM
		try:
			response = client.chat.completions.create(
				model=config.vl_model,
				messages=[
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
											背景信息：
											1. 所属文档标题："{md_file_name}"
											2. 图片上下文：
											   {context_info}
											请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
											生成一个精准的中文标题（不要包含"图片"二字）。""",
							},
							{
								"type": "image_url",
								"image_url": {
									"url": base64_image_data_url
								}
							}
						]
					}
				],
				temperature=0.3
			)
			summary = response.choices[0].message.content.strip()
			return summary
		except Exception as e:
			self.logger.info(f"图片摘要生成失败: {e}")
			return "图片描述"
	
	def _image_to_base64_data_url(self, image_path: str):
		# 读取文件的MIME类型
		mime_type, _ = mimetypes.guess_type(image_path)
		if not mime_type:
			return "图片格式MIME不正确"
		
		# 读取文件转化为base64
		try:
			with open(image_path, "rb") as image_file:
				base64_image = base64.b64encode(image_file.read()).decode("utf-8")
				return f"data:{mime_type};base64,{base64_image}"
		except IOError as e:
			self.logger.error(f"读取本地图片文件{image_path}失败: {e}")
			return "图片读取失败"
	
	def _enforce_rate_limit(self, request_timestamps: Deque[float], max_request: int, window_seconds: int = 60):
		"""
		
		Args:
			request_timestamps: 存放时间戳队列
			max_request: 窗口内最大请求数。
			window_seconds: 时间窗口大小（秒）。

		Returns:

		"""
		current_time = time.time()
		
		# 每次请求发起前先判断窗口中是否有已经超出时间窗口大小请求（清理🧹）
		while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
			request_timestamps.popleft()
		
		# 每次请求发起前判断是否超出当前时间窗口的最大请求数量
		if len(request_timestamps) >= max_request:
			sleep_duration = window_seconds - (current_time - request_timestamps[0])
			if sleep_duration >= 0:
				self.logger.info(f"当前时间窗口{window_seconds}内已达到最大API请求数量，暂停等待{sleep_duration:.2f}秒")
				time.sleep(sleep_duration)
			
			# 休眠完成之后再次清理超出时间窗口的请求
			current_time = time.time()
			while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
				request_timestamps.popleft()
		
		# 既未超出时间窗口 也未超出最大请求数量 直接入队列 视为一次请求
		request_timestamps.append(current_time)
	
	def _upload_images_to_minio(self, target_images_list, minio_client: Minio, md_path_obj: Path, config: ImportConfig):
		"""
		上传文件到Minio服务器
		Args:
			target_images_list: 目标图片
			minio_client: Minio客户端
			md_path_obj: Path对象
		Returns:
			remote_image_urls: Dict[图片名称，远程地址]
		"""
		remote_image_urls: Dict[str, str] = {}
		if not minio_client:
			self.logger.warn(f"Minio客户端未创建，无法上传本地文件至Minio服务器")
		
		for image_item in target_images_list:
			image_name, image_path, _ = image_item
			object_name = f"{md_path_obj.stem}/{image_name}".replace(" ","_")
			try:
				minio_client.fput_object(
					bucket_name=config.minio_bucket,
					object_name=object_name,
					file_path=image_path
				)
				remote_url = f"{config.get_minio_base_url()}/{object_name}"
				self.logger.info(f"图片上传成功: {image_name}")
				remote_image_urls[image_name] = remote_url
			except Exception as e:
				self.logger.error(f"图片上传失败: {image_name}:{e}")
				remote_image_urls[image_name] = ''
		
		return remote_image_urls
	
	def _replace_summary_and_remote_url(self, remote_image_urls, images_summarise, md_path_obj: Path, md_content: str):
		"""
		替换旧MD文档中图片的摘要和远程地址
		Args:
			remote_image_urls:
			images_summarise:

		Returns:

		"""
		new_md_content = md_content
		for image_name, image_summary in images_summarise.items():
			remote_url = remote_image_urls.get(image_name, "")
			if not remote_url:
				continue
			
			image_in_markdown_pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)", re.IGNORECASE)
			new_md_content = image_in_markdown_pattern.sub(f"![{image_summary}]({remote_url})", new_md_content)
		self.logger.info(f"成功替换{len(remote_image_urls)}张图片链接")
		return new_md_content


if __name__ == "__main__":
	# 测试用例
	setup_logging()
	imageNode = MDImageNode()
	imageNode.process({
		"md_path": "/Users/artest/Desktop/shopkeeper/output/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md"
	})
