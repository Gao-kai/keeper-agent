
from langgraph.graph import StateGraph,START,END

from knowledge.processor.query_process.nodes.confirm_item_name_node import ConfirmItemNameNode
from knowledge.processor.query_process.state import QueryGraphState


def route_condition(state:QueryGraphState):
	return bool(state.get("answer"))

def create_query_graph():
	# 1. 定义图编排工作流
	workflow = StateGraph(QueryGraphState)  # type:ignore
	
	# 2. 定义图节点
	nodes = {
		"confirm_item_name_node": ConfirmItemNameNode(),
	}
	
	# 3. 添加图节点
	for name, node in nodes.items():
		workflow.add_node(name, node)
		
	# 定义边
	workflow.add_edge(START,"confirm_item_name_node")
	workflow.add_conditional_edges(
		"confirm_item_name_node",
		route_condition,
		path_map={
			False:"multi_search",
			True: "answer_output"
		}
	)
	
	return workflow.compile()


if __name__ == "__main__":
	print("开始执行查询流程")
