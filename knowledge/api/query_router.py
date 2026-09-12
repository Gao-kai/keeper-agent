import logging
import os.path
from typing import Annotated
from fastapi import Depends, UploadFile, HTTPException
from fastapi import BackgroundTasks, Request
from starlette.responses import FileResponse, StreamingResponse

from knowledge.schema.query_schema import (
    QueryRequest,
    StreamSubmitResponse,
    QueryResponse,
)
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse
from knowledge.services.import_file_service import ImportFileService
from knowledge.services.query_service import QueryService
from knowledge.services.task_service import TaskService
from knowledge.utils.depends import (
    get_import_file_service,
    get_task_service,
    get_query_service,
)
from knowledge.utils.path_utils import get_static_dir_name
from knowledge.utils.sse_push import sse_generator


def create_query_api_route(app):
    """
    1. 普通API请求
    2. 静态资源请求
    3. 前端构建打包资源挂载静态资源服务器

    前端构建chat页面 就是首页
    点击去上传文档 跳转到上传页面
    所以这个后端应该支持首页的跳转
    """

    @app.get("/chat")
    async def chat_page():
        front_static_dir_name = get_static_dir_name()
        return FileResponse(path=os.path.join(front_static_dir_name, "index.html"))

    # 用户对话接口
    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        query_service: Annotated[QueryService, Depends(get_query_service)],
    ):
        session_id = request.session_id or query_service.generate_session_id()
        task_id = query_service.generate_task_id()
        is_stream = request.is_stream
        question = request.query
        # 1. 如果是流式输出 需要为当前task_id 也就是当前这个问答一个SSE队列
        # 2. 需要更新当前任务状态为处理中 `processing`
        query_service.submit_query_task(task_id, request.is_stream)

        # 3. 如果是流式输出: 异步执行query_service.run_query_graph 开始挨个节点执行
        if is_stream:
            background_tasks.add_task(
                query_service.run_query_graph, task_id, session_id, question, is_stream
            )
            return StreamSubmitResponse(
                message="查询任务已开始运行，请连接SSE进行流式输出",
                session_id=session_id,
                task_id=task_id,
            )
        else:
            # 4. 如果是普通输出: 执行图 返回答案
            query_service.run_query_graph(task_id, session_id, question, is_stream)
            answer = query_service.get_answer(task_id)
            return QueryResponse(
                message="查询任务处理完成", session_id=session_id, answer=answer
            )

    # SSE连接进行流式输出
    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        query_service: QueryService = Depends(get_query_service),
    ):
        try:
            items = query_service.get_history_message(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        query_service: QueryService = Depends(get_query_service),
    ):
        count = query_service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}

    @app.get("/health")
    async def health():
        return {"ok": True}
