import asyncio
import json
import logging
import queue
from typing import Dict, Optional, Any

from fastapi import Request

logger = logging.getLogger("SSE PUSH")


class SSEEvent:
    """SSE 事件类型常量。

    对应 SSE 协议里 `event:` 字段的取值，前端按事件名注册不同的监听回调。

    Attributes:
        READY: 连接就绪。生成器启动时立即下发，用于告知前端"SSE 通道已建立，
            可以开始渲染进度条了"，同时有助于穿透部分反向代理的响应缓冲。
        PROGRESS: 进度事件。由后台任务节点推送，data 一般携带当前节点名、
            已完成/运行中的节点列表等进度信息。
        DELTA: 增量文本事件。大模型流式生成时逐块推送，data 携带本块新增文本。
        FINAL: 结束事件。data 携带最终结果（如完整答案），前端收到后关闭连接。
    """

    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"


sse_task_bucket: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的 SSE 队列。

    Args:
        task_id: 任务 ID。

    Returns:
        该任务对应的队列；不存在时返回 None。

    Note:
        这是一个普通字典的直读，task_id 不存在会抛 KeyError。调用方通常应
        先经 create_sse_queue 创建，或用 `sse_task_bucket.get(task_id)` 语义访问。
    """
    return sse_task_bucket[task_id]


def create_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """为指定任务创建并注册 SSE 队列。

    应在启动后台任务之前调用，保证任务侧 push_sse_event 时队列已存在。

    Args:
        task_id: 任务 ID。

    Returns:
        新建的队列实例。

    Note:
        若 task_id 已存在，旧队列会被覆盖（原队列的待推消息随之丢失）。
    """
    task_queue = queue.Queue()
    sse_task_bucket[task_id] = task_queue
    return task_queue


def remove_sse_queue(task_id: str):
    """移除指定任务的 SSE 队列，释放内存。

    Args:
        task_id: 任务 ID。不存在时静默跳过。
    """
    sse_task_bucket.pop(task_id, None)


def pack_sse_message(event: str, data: Dict[str, Any]) -> str:
    """把事件与数据打包成符合 SSE 协议的字符串。

    产出格式::

        event: progress
        data:{"node":"re_rank_node"}

        （末尾空行表示本条消息结束）

    Args:
        event: 事件名，取 SSEEvent 中的常量。
        data: 事件数据，会被 JSON 序列化。

    Returns:
        可直接写入响应体的 SSE 报文字符串。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata:{payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """向指定任务推送一条 SSE 事件（供后台任务/图节点调用）。

    生产者侧入口：运行在后台线程或线程池中，只做入队，不感知连接状态。

    Args:
        task_id: 任务 ID。
        event: 事件名，取 SSEEvent 中的常量。
        data: 事件数据。

    Note:
        队列不存在时静默丢弃，不会抛异常（避免因为前端没连上而拖垮主流程）。
    """
    task_queue = get_sse_queue(task_id)
    if task_queue:
        task_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request):
    """SSE 响应生成器（异步生成器，作为 StreamingResponse 的 content 使用）。

    职责：把 sse_task_bucket 中该任务的队列消息，持续转换成 SSE 报文推给前端，
    直到客户端断开连接。它是"队列 → HTTP 流"的桥接层。

    典型用法::

        @app.get("/stream/{task_id}")
        async def stream(task_id: str, request: Request):
            return StreamingResponse(
                sse_generator(task_id, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    执行流程:

        1. 取出该任务的队列；队列不存在说明任务未创建或已清理，直接 return，
           StreamingResponse 会返回空响应体。
        2. 先 yield 一条 READY 事件，让前端确认通道已建立。
        3. 进入轮询循环:
           a. request.is_disconnected()
              检测客户端是否已断开（关闭页面/刷新/网络中断），断开则跳出循环结束流；

           b. 通过 asyncio.to_thread(task_queue.get, True, 1.0)
              类似JS中使用new Promise包装了一个异步执行的函数task_queue.get
              task_queue.get的意思就是去队列头部取出第一个元素并返回
              参数True的意思是当队列为空的时候等待 不会理解抛出队列Empty
              参数1.0的意思是等待1s的时间

              这样做的好处是避免task_queue.get执行的时候阻塞事件循环
              使循环每隔 1 秒就能重新检查一次连接状态；

           c. 取到消息后打包成 SSE 报文 yield 出去
        4. 无论正常结束还是异常退出，finally 中都会移除队列，防止内存泄漏。

    Args:
        task_id: 任务 ID，用于定位对应的消息队列。
        request: FastAPI/Starlette 请求对象，用于检测客户端是否断开连接。

    Yields:
        str: 符合 SSE 协议格式的报文，由 pack_sse_message 生成。

    Note:
        - 本生成器不主动发送 FINAL 事件也不主动结束：循环是"无限"的，
          真正的终止依赖客户端断开或任务侧推完后的连接关闭。若需要主动收尾，
          可在任务结束时 push 一条 FINAL 事件，并让前端收到后关闭 EventSource；
          或在队列里放入哨兵值（如 None）令循环 break。
        - asyncio.CancelledError 必须原样 re-raise，否则会破坏 asyncio 的
          取消语义（如 uvicorn 关闭连接时的清理流程）。
    """
    task_queue = get_sse_queue(task_id)

    if not task_queue:
        return

    try:

        while True:
            if await request.is_disconnected():
                break

            try:
                # 不停的将队列中的数据推给前端 除非1s之内队列还没有数据就挂起
                msg = await asyncio.to_thread(task_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            yield pack_sse_message(event, data)
    except (ConnectionError, BrokenPipeError):
        # 客户端刷新页面或关闭标签页 TCP连接断开 静默退出
        logger.warning("客户端刷新页面或关闭标签页 TCP连接断开 静默退出")
    except asyncio.CancelledError:
        # asyncio.CancelledError 必须原样 re-raise，否则会破坏 asyncio 的取消语义
        raise
    finally:
        # 清理资源（防止内存泄漏）
        remove_sse_queue(task_id)
