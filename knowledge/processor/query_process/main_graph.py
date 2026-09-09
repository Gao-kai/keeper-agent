import json

from langgraph.graph import StateGraph, START, END

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.confirm_item_name_node import (
    ConfirmItemNameNode,
)
from knowledge.processor.query_process.nodes.hybrid_vector_search_node import (
    HybridVectorSearchNode,
)
from knowledge.processor.query_process.nodes.hyde_document_search_node import (
    HydeDocumentEmbeddingSearchNode,
)
from knowledge.processor.query_process.nodes.query_knowledge_graph_node import (
    QueryKnowledgeGraphNode,
)
from knowledge.processor.query_process.nodes.re_rank_node import ReRankNode
from knowledge.processor.query_process.nodes.rrf_rank_node import RRFRankNode
from knowledge.processor.query_process.nodes.web_search_mcp_node import WebSearchMCPNode
from knowledge.processor.query_process.state import (
    QueryGraphState,
    create_default_state,
)
from knowledge.utils.log_config import setup_logging


def route_condition(state: QueryGraphState):
    return bool(state.get("answer"))


def create_query_graph():
    """创建查询流程图。

    注意：
    1. LangGraph中每个节点的返回值都是增量更新，类似Object.Assign的作用
    2. 因此在并行执行的节点中不能多个节点同时操作相同的字段，否则会并行更新异常，推荐只更新当前节点相关的字段
    3. 汇总节点的功能是防止某一路节点完成之后立即进入rrf节点触发节点执行
    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        item_name_confirm
              │
              ├── (有答案) ──────────────────────────> answer_output
              │                                              │
              └── (无答案) ──> multi_search ─────┬──────────>│
                                   │             │           │
                         ┌─────────┼─────────────┼───────┐   │
                         │         │             │       │   │
                         v         v             v       v   │
                   embedding  hyde_embedding  query_kg  web  │
                         │         │             │       │   │
                         └─────────┴─────────────┴───────┘   │
                                       │                     │
                                       v                     │
                                     join                    │
                                       │                     │
                                       v                     │
                                      rrf                    │
                                       │                     │
                                       v                     │
                                    rerank                   │
                                       │                     │
                                       v                     │
                               answer_output <───────────────┘
                                       │
                                       v
                                      END
    """

    # 1. 定义图编排工作流
    workflow = StateGraph(QueryGraphState)  # type: ignore

    # 2. 定义图节点
    nodes = {
        "confirm_item_name_node": ConfirmItemNameNode(),
        "multi_search_node": lambda x: {},  # 多路搜索分发（虚节点）
        "hybrid_vector_search_node": HybridVectorSearchNode(),
        "hyde_document_embedding_search_node": HydeDocumentEmbeddingSearchNode(),
        "query_knowledge_graph_node": QueryKnowledgeGraphNode(),
        "web_search_mcp_node": WebSearchMCPNode(),
        "search_join_node": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf_rank_node": RRFRankNode(),
        "re_rank_node": ReRankNode(),
        "answer_output_node": AnswerOutputNode(),
    }

    # 3. 添加图节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 定义边
    workflow.add_edge(START, "confirm_item_name_node")
    workflow.add_conditional_edges(
        "confirm_item_name_node",
        route_condition,
        path_map={False: "multi_search_node", True: "answer_output_node"},
    )

    # 多路检索（并行执行）
    workflow.add_edge("multi_search_node", "hybrid_vector_search_node")
    workflow.add_edge("multi_search_node", "hyde_document_embedding_search_node")
    workflow.add_edge("multi_search_node", "query_knowledge_graph_node")
    workflow.add_edge("multi_search_node", "web_search_mcp_node")

    # 多路检索结果汇总
    workflow.add_edge("hybrid_vector_search_node", "search_join_node")
    workflow.add_edge("hyde_document_embedding_search_node", "search_join_node")
    workflow.add_edge("query_knowledge_graph_node", "search_join_node")
    workflow.add_edge("web_search_mcp_node", "search_join_node")

    # 添加顺序边
    workflow.add_edge("search_join_node", "rrf_rank_node")
    workflow.add_edge("rrf_rank_node", "re_rank_node")
    workflow.add_edge("re_rank_node", "answer_output_node")

    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


def run_query_graph(
    question: str, session_id: str, item_names: list = None, is_stream: bool = False
):
    """

    Args:
        question:
        session_id:
        item_names:
        is_stream:

    Returns:

    """

    # 1. 创建全局图实例
    query_graph = create_query_graph()
    print(f"查询流程图示意图\n")
    query_graph.get_graph().print_ascii()

    # 2. 创建初始状态
    initial_state = create_default_state(
        session_id=session_id or "default",
        original_query=question,
        item_names=item_names or [],
        is_stream=is_stream,
    )

    # 3. 执行调用返回图更新后最新的state
    final_state = None
    for event in query_graph.stream(initial_state):  # type: ignore
        for node_name, state in event.items():
            print(f"✅✅✅ 当前执行节点{node_name} ✅✅✅")
            final_state = state

    print(f"流程执行完成: {json.dumps(final_state, ensure_ascii=False, indent=4)}")


if __name__ == "__main__":
    # 日志配置
    setup_logging()
    print("开始执行查询流程")

    # 开始测试
    run_query_graph(
        session_id="0001",
        question="HUAWEI MateStation12电脑如何一键恢复出厂设置呢？",
        item_names=["HUAWEI MateStation S 12代酷睿版"],
        is_stream=False,
    )
