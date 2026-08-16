from knowledge.utils.task_status import (
	get_task_status,
	get_running_task_list,
	get_completed_task_list,
	add_running_task,
	add_completed_task,
	update_task_status)


class TaskService:
	
	@staticmethod
	def mark_node_running(task_id: str, node_name: str):
		add_running_task(task_id, node_name)
	
	@staticmethod
	def mark_node_done(task_id: str, node_name: str):
		add_completed_task(task_id, node_name)
	
	@staticmethod
	def update_task_status(task_id: str, status: str):
		update_task_status(task_id, status)
	
	@staticmethod
	def get_task_status(task_id: str):
		return get_task_status(task_id)
	
	def get_task_info(self, task_id: str):
		return {
			"status": self.get_task_status(task_id),
			"completed_list": get_completed_task_list(task_id),
			"running_list": get_running_task_list(task_id),
		}
