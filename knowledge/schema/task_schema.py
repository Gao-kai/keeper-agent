from typing import List

from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
	status:str = Field(...,description="任务状态")
	completed_list:List[str] = Field(...,description="已经完成任务节点列表")
	running_list:List[str] = Field(...,description="正在运行任务节点列表")