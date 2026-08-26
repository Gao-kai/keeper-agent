from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from knowledge.utils.log_config import setup_logging
from knowledge.processor.import_process.nodes.chunk_embedding_node import ChunkEmbeddingNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.item_node_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.import_knowledge_graph_node import KnowledgeGraphNode
from knowledge.processor.import_process.nodes.md_img_node import MDImageNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PDFToMarkDownNode
from knowledge.processor.import_process.nodes.save_to_milvus_node import SaveToMilvusNode
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
        "md_image_node": MDImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node":ItemNameRecognitionNode(),
        "chunk_embedding_node_name":ChunkEmbeddingNode(),
        "save_to_milvus_node":SaveToMilvusNode(),
        "knowledge_graph_node":KnowledgeGraphNode()
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
    graph_buildr.add_edge("md_image_node", "document_split_node")
    graph_buildr.add_edge("document_split_node", "item_name_recognition_node")
    graph_buildr.add_edge("item_name_recognition_node", "chunk_embedding_node_name")
    graph_buildr.add_edge("chunk_embedding_node_name", "save_to_milvus_node")
    graph_buildr.add_edge("save_to_milvus_node", "knowledge_graph_node")
    graph_buildr.add_edge("knowledge_graph_node", END)

    # 编译
    return graph_buildr.compile()

def run_import_graph(import_file_path:str,file_dir:str):
    # 1. 获取图的示意图
    import_graph = create_import_graph()
    print(f"流程图示意图\n")
    import_graph.get_graph().print_ascii()
    
    # 2. 构建初始状态
    import_file_path = import_file_path or "/Users/artest/Desktop/shopkeeper/data/doc/HUAWEI MateStation S 12代酷睿版 用户指南-(PUC,Windows11_02,zh-cn).pdf"
    file_dir = file_dir or "/Users/artest/Desktop/shopkeeper/output"
    init_state = create_default_state(**{
        "import_file_path": import_file_path,
        "file_dir": file_dir
    })
    
    # 3. 执行调用返回图更新后最新的state
    final_state = None
    for event in import_graph.stream(init_state):  # type: ignore
        for node_name,state in event.items():
            print(f"✅✅✅ 当前执行节点{node_name} ✅✅✅")
            final_state = state

    # print(f"流程执行完成: {json.dumps(final_state, ensure_ascii=False, indent=4)}")


if __name__ == "__main__":
    # 日志配置
    setup_logging()
    
    # 开始测试
    run_import_graph()
    
   

   