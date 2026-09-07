from typing import List

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.config import get_query_config, QueryConfig
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        答案生成节点
        1. 检查是否已有答案，如果有直接返回（商品名有多个或者无商品名时）
        2. 构建提示词模版
        3. 调用LLM
        4. 写入历史会话记录
        5. 发送结束事件

        Args:
                state:

        Returns:

        """
        config = get_query_config()

        answer = state.get("answer")
        if answer:
            return state
        else:

            is_stream = state.get("is_stream")
            prompt = self.build_prompt(state, config)
            state["prompt"] = prompt
            self.generate_answer(state, prompt)
        return state

    def build_prompt(self, state: QueryGraphState, config: QueryConfig):
        """
        构建提示词模版

        顺序：
        1. 系统提示词
        2. re-rank重排序信息
        3. 历史消息
        4. neo-4j三元组结构化信息
        5. 重写后用户问题

        字符预算控制：
        最大不能超出config中配置的最大字符
        Args:
            state:
            config:

        session_id: 会话 ID，用于追踪多轮对话。
        message_id: 消息 ID，标识单次查询。
        original_query: 原始用户查询。
        embedding_chunks: 向量检索结果列表。
        hyde_embedding_chunks: HyDE 检索结果列表。
        rrf_chunks: RRF 融合后的切片列表。
        web_search_docs: 网页搜索结果列表。
        reranked_docs: 重排序后的文档列表。
        prompt: 构造的提示词。
        answer: 最终生成的答案。
        item_names: 识别的商品名称列表。
        rewritten_query: 重写后的查询。
        history: 历史对话列表。
        is_stream: 是否启用流式输出。
        graph_chunks: 知识图谱相关切片列表。
        graph_relation_texts: 知识图谱三元组列表。

        Returns:

        """

        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names")
        available_llm_prompt_length = config.max_context_chars

        re_ranked_doc_prompts, available_llm_prompt_length = (
            self.generate_re_ranked_doc_prompt(
                state, config, available_llm_prompt_length
            )
        )
        graph_relation_texts_prompts, available_llm_prompt_length = (
            self.generate_graph_relation_texts_prompt(
                state, config, available_llm_prompt_length
            )
        )

        return ANSWER_PROMPT.format(
            re_ranked_docs=re_ranked_doc_prompts,
            history="暂无历史对话",
            item_names=item_names,
            graph_relation_texts=graph_relation_texts_prompts,
            question=question,
        )

    @staticmethod
    def generate_re_ranked_doc_prompt(
        state: QueryGraphState,
        config: QueryConfig,
        available_llm_prompt_length: int,
    ):
        """
        基于re-ranked_docs构建提示词
        Args:
            state:
            config:
            available_llm_prompt_length

        Returns:

        """
        if available_llm_prompt_length <= 0:
            return None

        reranked_docs = state.get("reranked_docs")
        if not reranked_docs:
            return None

        prompt_list = []
        for index, reranked_doc in enumerate(reranked_docs):
            collection: List[str] = [f"[{index+1}]"]
            content = reranked_doc.get("content")
            for key, value in reranked_doc.items():
                if key == "content":
                    continue

                if not value:
                    value = "None"

                collection.append(f"[{key}={value}]")

            reranked_doc_prompt = f"""
{" ".join(collection)}\n
{content}""".strip()
            prompt_list.append(reranked_doc_prompt)

        re_rank_doc_prompts = "\n\n".join(prompt_list)
        prompt_len = len(re_rank_doc_prompts)

        if available_llm_prompt_length - prompt_len <= 0:
            return None

        return re_rank_doc_prompts, available_llm_prompt_length

    @staticmethod
    def generate_graph_relation_texts_prompt(
        state: QueryGraphState,
        config: QueryConfig,
        available_llm_prompt_length: int,
    ):
        """
        基于Neo-4j结构化关系构建提示词
        每条三元组格式化为 "item_name head -[relation]-> tail"（无 item_name 时省略前缀）
        Args:
            state:
            config:
            available_llm_prompt_length:

        Returns:

        """
        if available_llm_prompt_length <= 0:
            return None

        graph_relation_texts = state.get("graph_relation_texts")

        if not graph_relation_texts:
            return None

        prompt_list = []
        for index, graph_relation_text in enumerate(graph_relation_texts):
            collection: List[str] = [f"[{index+1}]", f"{graph_relation_text}"]
            graph_relation_text_prompt = "  ".join(collection)
            prompt_list.append(graph_relation_text_prompt)

        graph_relation_text_prompts = "\n".join(prompt_list)
        prompt_len = len(graph_relation_text_prompts)

        if available_llm_prompt_length - prompt_len <= 0:
            return None

        return graph_relation_text_prompts, available_llm_prompt_length

    def generate_answer(self, state: QueryGraphState, prompt: str):
        """
        生成答案
        Args:
            state:
            prompt:

        Returns:

        """


if __name__ == "__main__":
    answerOutputNode = AnswerOutputNode()
    __state = {
        "reranked_docs": [
            {
                "chunk_id": "468308814313816314",
                "content": "## 用户指南\n\n了解计算机\n    外观介绍 1\n    键盘 2\n    开启和关闭计算机 3\n    F10 一键恢复出厂 3\n    获取精彩功能 3\n安全信息\n个人信息和数据安全\n法律声明\n\n![主机外观及接口说明](http://localhost:9000/knowledge-base/HUAWEI_MateStation_S_12代酷睿版_用户指南-(PUC,Windows11_02,zh-cn)/621be1d7752348735664e224925f70d7452e3f65643b55ea2507031c3dac50e4.jpg)",
                "score": 0.97192098251611,
                "source": "rrf",
                "title": "",
                "url": "",
            },
            {
                "chunk_id": "468308814313816319",
                "content": "## 电池安全\n\n• 如果更换不正确的型号的电池会有起火或爆炸的危险。\n\n• 请勿将电池暴露在高温处或发热产品的周围，如日照、取暖器、微波炉、烤箱或热水器等。电池过热可能引起爆炸。\n\n• 请勿将电池放置在极低气压环境中，可能导致电池爆炸或泄漏可燃液体或气体。",
                "score": 0.371987249157216,
                "source": "rrf",
                "title": "",
                "url": "",
            },
            {
                "chunk_id": "468308814313816322",
                "content": "## 个人信息和数据安全\n\n在使用设备的一些功能和第三方应用时，可能会因为操作不正确或其他原因导致您的个人信息或数据泄露或丢失，建议按以下方式加强保护您的个人信息。\n\n• 请将设备放置于安全区域，防止未经授权人员使用您的设备。\n\n• 建议不要阅读来自陌生人的信息或邮件，以免设备遭受病毒感染。",
                "score": 0.11355395814653775,
                "source": "rrf",
                "title": "HUAWEI MateStation S 12代酷睿版 用户指南-(PUC,Windows11_02,zh-cn)",
                "url": "",
            },
        ],
        "graph_relation_texts": [
            "HUAWEI MateStation S 12代酷睿版 设备 -(HAS_PART)-> 电源",
            "HUAWEI MateStation S 12代酷睿版 个人信息和数据安全 -(HAS_STEP)-> 步骤1-放置安全区域",
            "HUAWEI MateStation S 12代酷睿版 步骤1-放置安全区域 -(NEXT_STEP)-> 步骤2-勿读陌生信息",
        ],
        "rewritten_query": "HUAWEI MateStation12电脑使用电源有哪些需要注意的安全点？",
        "item_names": ["HUAWEI MateStation S 12代酷睿版"],
    }

    answerOutputNode.process(__state)
