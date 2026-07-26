import json
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_process.log_config import setup_logging
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_img_node import MDImageNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PDFToMarkDownNode
from knowledge.processor.import_process.state import ImportGraphState, create_default_state

def route_condition(state: ImportGraphState) -> Literal["pdf", "md"]:
    """
    路由条件函数，根据文件类型判断是否需要处理为md文档
    """
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    elif state.get("is_md_read_enabled"):
        return "md"
    else:
        raise ValueError(f"不支持的文件类型: {state.get("import_file_path")}")


def create_import_graph() -> CompiledStateGraph:
    """
    基于langgraph创建workflow流程图

    Returns:
        编译后的 StateGraph 实例

    """
    # 1. 定义图工作流
    graph_buildr = StateGraph(ImportGraphState)  # type:ignore

    # 2. 定义图节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PDFToMarkDownNode(),
        "md_image_node": MDImageNode()
    }

    # 3. 添加图节点
    for name, node in nodes.items():
        graph_buildr.add_node(name, node)

    # 4. 定义边
    graph_buildr.add_edge(START, "entry_node")
    graph_buildr.add_conditional_edges(
        "entry_node",
        route_condition,
        {
            "pdf": "pdf_to_md_node", # 首先处理为md文档 再去处理md文档中image节点
            "md": "md_image_node" # 直接处理md文档中image节点
        }
    )
    graph_buildr.add_edge("pdf_to_md_node", "md_image_node")
    graph_buildr.add_edge("md_image_node", END)

    # 编译
    return graph_buildr.compile()


if __name__ == "__main__":
    # 日志配置
    setup_logging()
    
    # 获取图的示意图
    import_graph = create_import_graph()
    print(f"流程图示意图\n")
    import_graph.get_graph().print_ascii()

    # 构建图初始状态
    import_file_path = "/Users/artest/Desktop/shopkeeper/data/doc/H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
    file_dir = "/Users/artest/Desktop/shopkeeper/output"
    init_state = create_default_state(**{
        "import_file_path": import_file_path,
        "file_dir": file_dir
    })

    # 执行调用返回图更新后最新的state
    result = import_graph.invoke(init_state)  # type: ignore
    print(f"流程执行完成: {json.dumps(result, ensure_ascii=False, indent=4)}")
