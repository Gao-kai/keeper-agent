import logging
import os.path
import shutil
import uuid
from datetime import datetime
from os import makedirs
from typing import Tuple

from fastapi import UploadFile

from knowledge.processor.import_process.config import get_import_config
from knowledge.processor.import_process.exception import FileProcessingError
from knowledge.processor.import_process.main_graph import create_import_graph
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.main_graph import create_import_graph
from knowledge.processor.import_process.state import create_default_state
from knowledge.services.task_service import TaskService
from knowledge.utils.minio_client import get_minio_client
from knowledge.utils.path_utils import get_local_cache_dir_name
from dotenv import load_dotenv

from knowledge.utils.task_status import TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED

load_dotenv(override=True)

class ImportFileService:
	
	def __init__(self, task_service: TaskService):
		self.task_service = task_service
		self.logger = logging.getLogger(f"import.upload_file_node")
		self.config = get_import_config()
	
	def process_upload_file(self, file: UploadFile)->Tuple[str,str,str]:
		"""
		
		Returns:
		1. 标记upload_file_node节点为运行中
		2. 本地上传
		3. Minio上传
		4. 标记upload_file_node节点为结束
		5. 返回启动Graph的初始State
		"""
		
		# 1. 构建本地缓存日期目录
		upload_date = datetime.now().strftime("%Y-%m-%d")
		local_cache_dir = get_local_cache_dir_name()
		upload_date_dir = os.path.join(local_cache_dir, upload_date)
		
		# 2. 获取TASK_ID
		task_id = str(uuid.uuid4())
		self.task_service.mark_node_running(task_id,node_name="upload_file_node")
		
		# 3. 构建本地文件缓存目录（Graph State初始化参数）
		file_dir = os.path.join(upload_date_dir, task_id)
		
		# 4. 缓存上传文件至本地缓存目录并返回缓存好的本地文件路径
		import_file_path = self.save_file_to_local(file_dir, file)
		
		# 5. 将本地文件上传至Minio文件服务器
		object_name = self.save_file_to_minio(import_file_path,file)
		self.task_service.mark_node_done(task_id, node_name="upload_file_node")
		
		# 6. 返回参数
		return task_id,import_file_path,file_dir
	
	def run_import_graph(self,import_file_path:str, file_dir:str, task_id:str):
		# 更新当前任务状态为处理中
		self.task_service.update_task_status(task_id,TASK_STATUS_PROCESSING)
		
		# 获取图Graph
		import_graph = create_import_graph()
		print(f"流程图示意图\n")
		import_graph.get_graph().print_ascii()
		
		# 构建初始状态
		init_state = create_default_state(**{
			"task_id": task_id,
			"import_file_path": import_file_path,
			"file_dir": file_dir
		})
		
		# 执行调用返回图更新后最新的state
		final_state = None
		for event in import_graph.stream(init_state):  # type: ignore
			for node_name, state in event.items():
				print(f"✅✅✅ 当前执行节点{node_name} ✅✅✅")
				final_state = state
		# 更新当前任务状态为完成
		self.task_service.update_task_status(task_id,TASK_STATUS_COMPLETED)
	
	# print(f"流程执行完成: {json.dumps(final_state, ensure_ascii=False, indent=4)}")
	
	@staticmethod
	def save_file_to_local(file_dir, file: UploadFile):
		"""
		缓存上传文件至本地缓存目录
		Args:
			file_dir:
			file:

		Returns:
		"""
		
		if file is None:
			raise FileProcessingError(node_name="upload_file_node", message="文件不能为空")
		
		# 确保文件目录存在 exist_ok=True 即使存在也不报错
		makedirs(file_dir, exist_ok=True)
		
		# 获取完整文件保存路径
		import_file_path = os.path.join(file_dir, file.filename)
		
		# 保存文件
		with open(import_file_path, "wb") as f:
			# 分块复制，内存友好
			shutil.copyfileobj(file.file, f)
		
		return import_file_path
	
	def save_file_to_minio(self, import_file_path, file: UploadFile):
		
		minio_client = get_minio_client()
		if not minio_client:
			self.logger.warn(f"Minio客户端未创建，无法上传本地文件至Minio服务器")
		
		# 文件名
		filename = file.filename
		
		# 要上传到Minio服务器的对象名
		object_name = f"origin_files/{datetime.now().strftime("%Y-%m-%d")}/{filename}".replace(" ", "_")
		try:
			minio_client.fput_object(
				bucket_name=os.getenv("MINIO_BUCKET_NAME"),
				object_name=object_name,
				file_path=import_file_path
			)
			remote_url = f"{self.config.get_minio_base_url()}/{object_name}"
			self.logger.info(f"文件上传成功: {filename}")
			self.logger.info(f"云端预览地址: {remote_url}")
		
		except Exception as e:
			self.logger.error(f"文件上传失败: {filename}:{e}")
		
		return object_name


if __name__ == "__main__":
	print(datetime.now().strftime("%Y-%m-%d"))
