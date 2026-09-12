import logging
import os.path
import uvicorn
from typing import Annotated

from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks
from starlette.responses import FileResponse
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse
from knowledge.services.import_file_service import ImportFileService
from knowledge.services.task_service import TaskService
from knowledge.utils.depends import get_import_file_service, get_task_service
from knowledge.utils.path_utils import get_static_dir_name


def create_import_api_route(app):
    """
    1. 普通API请求
    2. 静态资源请求
    3. 前端构建打包资源如何挂载对静态资源服务器上
    4. 如何配置Nginx实例（真实打包挂载）
    """

    # 入口页面静态资源请求
    @app.get("/upload")
    async def get_index():
        front_static_dir_name = get_static_dir_name()
        return FileResponse(path=os.path.join(front_static_dir_name, "index.html"))

    # 上传接口
    @app.post("/upload_file", response_model=UploadResponse)
    async def upload_file(
        file: UploadFile,
        import_file_service: Annotated[
            ImportFileService, Depends(get_import_file_service)
        ],  # Depends:依赖注入
        background_tasks: BackgroundTasks,
    ):
        try:
            # 1. 将用户上传的文件上传至Minio服务
            task_id, import_file_path, file_dir = (
                import_file_service.process_upload_file(file=file)
            )

            # 2. 开始节点编排流程（background_tasks:HTTP 响应先发给前端，耗时任务在响应发送之后继续执行。）
            background_tasks.add_task(
                import_file_service.run_import_graph,
                import_file_path,
                file_dir,
                task_id,
            )

            # 3. 返回响应结果
            return UploadResponse(message="文件上传成功", task_id=task_id)
        except Exception as e:
            logging.error(f"上传接口报错:{e}")
            return UploadResponse(message="文件上传失败", task_id="")

    # 基于task_id查询当前任务状态
    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(
        task_id: str, task_service: Annotated[TaskService, Depends(get_task_service)]
    ):
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)
