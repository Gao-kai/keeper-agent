"""
定义前端轮训查询当前文件处理状态的函数
"""
from collections import defaultdict
from typing import Dict, List

# 定义Task任务状态
task_status: Dict[str, str] = {}
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 当前正在运行的TASK列表（defaultdict函数表示创建一个字典Dict 当访问的key不存在的时候自动用一个list空列表作为key的默认值）
task_running_list: Dict[str, List[str]] = defaultdict(list)

# 当前已经完成的TASK列表
task_completed_list: Dict[str, List[str]] = defaultdict(list)

# 当前任务执行结果
task_result: Dict[str, Dict[str, str]] = defaultdict(dict)

# 节点中英文转换
_NODE_NAME_TO_CN: Dict[str, str] = {
	"upload_file": "上传文件",
	"entry_node": "检查文件",
	"pdf_to_md_node": "PDF转Markdown",
	"md_image_node": "Markdown图片处理",
	"document_split_node": "文档切分",
	"item_name_recognition_node": "主体名称识别",
	"chunk_embedding_node_name": "向量生成",
	"save_to_milvus_node": "导入向量数据库",
	"knowledge_graph_node": "导入知识图谱",
	"__end__": "处理完成"
}


def to_zh_cn(node_name: str) -> str:
	return _NODE_NAME_TO_CN.get("node_name", node_name)


def add_running_task(task_id: str, node_name: str):
	"""
	为task_id的任务添加当前运行的节点名称
	Args:
		task_id:
		node_name:

	Returns:

	"""
	current_running_list = task_running_list.get(task_id)
	# 避免重复添加
	if node_name not in current_running_list:
		current_running_list.append(node_name)


def add_completed_task(task_id: str, node_name: str):
	"""
	为task_id的任务添加已经运行完成的节点名称
	Args:
		task_id:
		node_name:

	Returns:

	"""
	# 如果当前节点在运行过程 移除
	current_running_list = task_running_list.get(task_id)
	if node_name in current_running_list:
		current_running_list.remove(node_name)
	
	current_completed_list = task_running_list.get(task_id)
	
	# 避免重复添加
	if node_name not in current_completed_list:
		current_completed_list.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
	# 1. 获取指定任务运行中的节点列表，并通过列表推导式统一转换为中文展示名返回
	return [to_zh_cn(n) for n in task_running_list.get(task_id, [])]


def get_completed_task_list(task_id: str) -> List[str]:
	# 1. 获取指定任务已完成的节点列表，并通过列表推导式统一转换为中文展示名返回
	return [to_zh_cn(n) for n in task_completed_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
	"""
	根据任务ID 获取任务状态
	:param task_id:
	:return:
	"""
	# 1. 安全获取指定任务的总体运行状态，若不存在则返回空字符串
	return task_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
	# 1. 更新指定任务的总体运行状态（如 pending, processing 等）
	task_status[task_id] = status_name


def clear_task(task_id: str):
	# 1. 安全移除该任务的运行节点记录
	task_running_list.pop(task_id, None)
	# 2. 安全移除该任务的已完成节点记录
	task_completed_list.pop(task_id, None)
	# 3. 安全移除该任务的总体状态记录
	task_status.pop(task_id, None)
	# 4. 安全移除该任务的结果记录
	task_result.pop(task_id, None)
