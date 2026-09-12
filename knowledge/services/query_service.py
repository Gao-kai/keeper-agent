import logging
import uuid

from knowledge.processor.query_process.config import get_query_config
from knowledge.processor.query_process.main_graph import run_query_graph
from knowledge.services.task_service import TaskService
from knowledge.utils.mongo_client import get_mongo_tool
from knowledge.utils.sse_push import create_sse_queue, push_sse_event, SSEEvent
from knowledge.utils.task_status import (
    update_task_status,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    get_task_status,
    get_completed_task_list,
    get_running_task_list,
    get_task_result,
)

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, task_service: TaskService):
        self.task_service = task_service
        self.logger = logging.getLogger(f"import.upload_file_node")
        self.config = get_query_config()

    @staticmethod
    def generate_session_id():
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id():
        return str(uuid.uuid4())

    @staticmethod
    def submit_query_task(task_id: str, is_stream: bool):
        """
        1. 如果是流式输出 需要为当前task_id 也就是当前这个问答一个SSE队列
        2. 需要更新当前任务状态为处理中 `processing`
        Args:
            task_id:
            is_stream:

        Returns:
        """
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    @staticmethod
    def run_query_graph(task_id: str, session_id: str, question: str, is_stream: bool):
        """
        运行LangGraph
        Args:
            task_id:
            session_id:
            question:
            is_stream:

        Returns:

        """
        try:
            run_query_graph(
                question,
                session_id,
                task_id,
                [],
                is_stream,
            )
        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
        finally:
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            if is_stream:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.PROGRESS,
                    data={
                        "status": get_task_status(task_id),
                        "done_list": get_completed_task_list(task_id),
                        "running_list": get_running_task_list(task_id),
                    },
                )

    @staticmethod
    def get_answer(task_id: str):
        return get_task_result(task_id)

    @staticmethod
    def get_history_message(session_id: str, limit: int):
        mongo_tool = get_mongo_tool()
        if mongo_tool is not None:
            chat_history = mongo_tool.get_history_message(
                session_id=session_id, limit=limit
            )

            return [
                {
                    "_id": str(item.get("_id", "")),
                    "session_id": item.get("session_id", ""),
                    "role": item.get("role", ""),
                    "text": item.get("text", ""),
                    "query": item.get("query", ""),
                    "item_names": item.get("item_names", []),
                    "timestamp": item.get("timestamp"),
                }
                for item in chat_history
            ]

        return []

    @staticmethod
    def clear_history(session_id: str) -> int:
        mongo_tool = get_mongo_tool()
        if mongo_tool is not None:
            return mongo_tool.clear_history(session_id)

        return 0
