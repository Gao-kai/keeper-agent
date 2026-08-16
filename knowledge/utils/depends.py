from anyio.functools import lru_cache

from knowledge.services.task_service import TaskService
from knowledge.services.import_file_service import ImportFileService

"""
缓存淘汰规则（LRU）
缓存满了（达到 maxsize）时，会淘汰最久没被使用的条目——如果某个结果长时间没被访问，下次新结果进来时它先被移除。
"""
@lru_cache(maxsize=128)
def get_task_service():
	return TaskService()

@lru_cache(maxsize=128)
def get_import_file_service():
	task_service = get_task_service()
	return ImportFileService(task_service=task_service)
