import datetime
import logging
import os
from typing import List

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

mongo_tool = None


class MongoDBTool:
    """
    MongoDB历史对话读写工具类
    """

    def __init__(self):
        try:
            self.mongo_uri = os.getenv("MONGO_URL")
            self.mongo_client = MongoClient(self.mongo_uri)
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.db = self.mongo_client[self.db_name]
            self.chat_message = self.db["chat_message"]
            logger.info(f"连接Mongo DB成功 ✅：{self.db_name}")
            # 创建索引Index加速查询
            self.chat_message.create_index([("session_id", 1), ("timestamp", -1)])
        except Exception as e:
            logger.error(f"连接Mongo DB失败 ❌：{self.db_name} -- 失败原因:{e}")
            raise

    def clear_history(self, session_id: str):
        """
        清空历史记录
        Returns:
        """
        try:
            result = self.chat_message.delete_many({"session_id": session_id})
            logger.info(f"删除 {result.deleted_count} 条消息成功 ✅：{session_id}")
            return result.deleted_count
        except Exception as e:
            logger.info(f"删除 {session_id} 所属消息失败 ❌：{e}")
            return 0

    def get_history_message(self, session_id: str, limit: int = 10):
        """
        查询历史会话记录（默认查询最近10条）
        Args:
            session_id:
            limit:

        Returns:
        1000
        1001
        1002
        1003

        """
        try:
            """
            1. 先取出所有数据（索引优化不会全部查）
            2. 将数据倒序排列
            3. 然后取出前4条
            4. 但是返回给大模型的不能是倒序的 还是得reverse一次 保证给大模型和用户交互顺序一致
            """
            cursor = (
                (self.chat_message.find({"session_id": session_id}))
                .sort([("timestamp", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            messages = list(cursor)
            messages.reverse()
            logger.info(f"查询 {session_id} 中最近{limit}条消息成功 ✅")
            return messages
        except Exception as e:
            logger.error(
                f"查询 {session_id} 中最近{limit}条消息失败 ❌ -- 失败原因:{e}"
            )
            return []

    def save_history_message(
        self,
        session_id: str,
        role: str,
        text: str,
        query: str,
        item_names: List[str] = None,
        message_id: str = None,
    ):
        """
        存储历史消息对话对所属session_id
        Args:
            session_id:
            role:
            text:
            query:
            item_names:
            message_id:主键

        Returns:

        """
        try:

            timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
            document = {
                "session_id": session_id,
                "role": role,
                "text": text,
                "query": query,
                "item_names": item_names,
                "timestamp": timestamp,
            }

            # 更新操作
            if message_id:
                self.chat_message.update_one(
                    filter={
                        "_id": ObjectId(message_id),
                    },
                    update={"$set": document},
                )
                logger.info(f"更新 {message_id} 消息成功 ✅")
                return message_id
            # 插入操作
            else:
                result = self.chat_message.insert_one(document)
                logger.info(f"插入 {result.inserted_id} 消息成功 ✅")
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"更新/新增消息失败 ❌：{e}")
            return None

    def update_message_item_names(self, ids: List[str], item_names: List[str]) -> int:
        """
        批量更新历史会话中的 item_names
        第一轮对话 大模型未识别出商品名 此时这一条消息对话中item_names为[]
        第二轮对话 大模型识别出商品名A和B 此时需要将当前这一个session中所有消息对话中item_names更新为[A,B]

        Args:
            ids:
            item_names:

        Returns:

        """
        try:
            object_ids = [ObjectId(id) for id in ids]
            result = self.chat_message.update_many(
                filter={
                    "_id": {"$in": object_ids},
                    # 「实际为空」才更新：不存在 / [] / null / [""] / [" "] 等
                    # 只要有任意一个非空白字符串元素，就说明已有商品名，跳过
                    "$nor": [{"item_names": {"$elemMatch": {"$regex": r"\S"}}}],
                },
                update={"$set": {"item_names": item_names}},
            )
            logger.info(f"更新 {result.modified_count} 条消息中的{item_names}成功 ✅")
        except Exception as e:
            logger.error(f"批量更新 item_names 失败 ❌：{e}")
            return 0


def get_mongo_tool() -> MongoDBTool | None:
    try:
        global mongo_tool

        if mongo_tool is not None and isinstance(mongo_tool, MongoDBTool):
            return mongo_tool

        mongo_tool = MongoDBTool()
        return mongo_tool
    except Exception as e:
        logger.error(f"创建Mongo DB工具类实例MongoDBTool报错: {e}")
        return None


if __name__ == "__main__":
    mongo_tool = get_mongo_tool()
