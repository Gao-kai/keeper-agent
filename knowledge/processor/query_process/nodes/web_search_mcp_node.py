"""
AI Agent是干嘛的（代理程序员干活的）
HTTP调用查询天气这个事情，只有两个路：
1. 程序员自己手动编码 自己调用axios.get
2. 程序员让Agent代替自己自动干 先创建agent 然后agent.invoke
3. 创建mcp客户端调用（因为去调用mcp服务端必须要用MCP协议 因此必须要创建MCP块和uu单）

### MCP 完整调用流程

---

#### 启动阶段（用户提问之前）

1. AI 应用启动，通过 MCP 协议连接预先配置好的 MCP Server
2. AI 应用向 MCP Server 发送 `tools/list` 请求，自动发现该 Server 提供的所有工具及其参数定义
3. AI 应用将获取到的工具列表注入大模型的 `tools` 参数中，使大模型"知道"自己拥有哪些可调用的能力

---

#### 用户提问 → 工具调用 → 最终回复

4. 用户向 AI 应用提问："今天上海天气多少度？"
5. AI 应用将用户问题连同工具列表一起发送给大模型
6. 大模型进行意图识别，判断需要调用工具，返回 Tool Message，包含工具名称 `query_weather` 和参数 `city: "上海"`
7. AI 应用拿到 Tool Message，通过 MCP 协议向对应的 MCP Server 发起 `tools/call` 请求
8. MCP Server 收到符合协议规范的调用请求，执行内部预定义的逻辑（如调用第三方天气 API）
9. MCP Server 将查询结果（如 `"28°C"`）通过 MCP 协议返回给 AI 应用
10. AI 应用将工具返回的结果与原始用户问题一起拼回上下文，再次发送给大模型
11. 大模型基于工具返回的真实数据，生成最终的自然语言回答，返回给用户

---

#### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      启动阶段（一次性）                        │
│                                                             │
│  AI应用 ──MCP: tools/list──→ MCP Server                     │
│  AI应用 ←──返回工具列表──────┘                               │
│  AI应用 ──将工具列表注入大模型 tools 参数──→ 大模型就绪        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      运行时（每次提问）                        │
│                                                             │
│  用户                                                       │
│   │ "今天上海天气多少度？"                                    │
│   ↓                                                         │
│  AI应用 ──用户问题 + 工具列表──→ 大模型                       │
│                                   │                         │
│                                   │ Tool Message:           │
│                                   │ 工具: query_weather     │
│                                   │ 参数: city="上海"        │
│                                   ↓                         │
│  AI应用 ←─── Tool Message ────────┘                         │
│   │                                                         │
│   │ MCP: tools/call                                         │
│   ↓                                                         │
│  MCP Server ──调用天气API──→ 第三方天气服务                   │
│   ↑                              │                          │
│   │         返回 "28°C"          │                          │
│   └──────────────────────────────┘                          │
│   │                                                         │
│   │ 将工具结果拼回上下文                                     │
│   ↓                                                         │
│  大模型 ──→ "今天上海28°C，天气晴" ──→ 用户                  │
└─────────────────────────────────────────────────────────────┘
```

### 数据接口

【启动阶段】

AI应用 ──→ MCP Server
{
  "jsonrpc": "2.0", "id": 1,
  "method": "tools/list",
  "params": { "cursor": "" }
}

AI应用 ←── MCP Server
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "tools": [{
      "name": "query_weather",
      "description": "查询指定城市的实时天气信息",
      "inputSchema": { ... }
    }]
  }
}

→ AI应用把 tools 注入大模型的 tools 参数，大模型就绪


【运行阶段】

用户："今天上海天气多少度？"
大模型决策 → Tool Message: { name: "query_weather", arguments: { city: "上海" } }

AI应用 ──→ MCP Server
{
  "jsonrpc": "2.0", "id": 2,
  "method": "tools/call",
  "params": {
    "name": "query_weather",
    "arguments": { "city": "上海" }
  }
}

AI应用 ←── MCP Server
{
  "jsonrpc": "2.0", "id": 2,
  "result": {
    "content": [{ "type": "text", "text": "28°C，晴" }],
    "isError": false
  }
}

→ AI应用把结果拼回上下文，大模型生成最终回答："今天上海28°C，天气晴"

"""
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult

from knowledge.processor.query_process.base import BaseNode, T, setup_logging
from knowledge.processor.query_process.config import get_query_config, QueryConfig
from knowledge.processor.query_process.exception import ValidationError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMCPNode(BaseNode):
	
	def process(self, state: QueryGraphState) -> QueryGraphState:
		# 1. 参数校验
		self.log_step("step_1", f"参数校验")
		query_config = get_query_config()
		item_names, rewritten_query = self.validate_inputs(state)
		if not rewritten_query:
			self.logger.warning("查询内容为空，跳过网络检索")
		
		# 2. 调用Open AI Agent SDK 创建MCP服务端
		self.log_step("step_2", f"创建MCP网络查询服务")
		try:
			tool_call_result = asyncio.run(self.mcp_server_call(query_config, rewritten_query))
			if tool_call_result:
				web_search_docs = self.format_tool_call_result(tool_call_result)
				self.log_step("step_5", f"搜索完成 总计{len(web_search_docs)}条结果")
				return {
					"web_search_docs": web_search_docs
				}
		except Exception as e:
			self.logger.error(f"MCP 搜索失败: {e}")
		
		return state
	
	async def mcp_server_call(self, query_config: QueryConfig, rewritten_query: str):
		"""

		Args:
			query_config:
			rewritten_query:

		Returns:

		"""
		# 1. 基于Streamable HTTP 创建MCP服务端
		search_mcp_server = MCPServerStreamableHttp(
			name="百炼通用搜索 MCP Server",
			params={
				"url": query_config.mcp_dashscope_base_url,
				"headers": {
					"Authorization": f"Bearer {query_config.openai_api_key}"
				}
			},
			cache_tools_list=True,
			max_retry_attempts=5
		)
		
		try:
			
			# 2. 连接
			await search_mcp_server.connect()
			
			# 3. 获取工具列表
			tool_list = await  search_mcp_server.list_tools()
			
			tool_list_names = "\n".join([tool.name for tool in tool_list])
			self.logger.info(f"连接MCP服务名称：{search_mcp_server.name}")
			self.logger.info(f"可用工具列表为:{tool_list_names}")
			
			# 4. 调用工具
			self.log_step("step_3", f"调用MCP网络查询服务")
			tool_call_result = await search_mcp_server.call_tool(
				tool_name="bailian_web_search",
				arguments={
					"query": rewritten_query,
					"count": 3
				}
			)
			
			return tool_call_result
		
		finally:
			# 清理资源
			await search_mcp_server.cleanup()
	
	def format_tool_call_result(self, tool_call_result: CallToolResult):
		if not tool_call_result:
			return []
		
		self.log_step("step_4", f"格式化MCP服务返回结果")
		web_search_docs = []
		
		content = tool_call_result.content
		text_content = content[0]
		text = text_content.text
		
		try:
			parsed_text_json = json.loads(text)
			pages = parsed_text_json.get("pages", [])
			
			for page in pages:
				snippet = (page.get("snippet") or "").strip()
				url = (page.get("url") or "").strip()
				title = (page.get("title") or "").strip()
				if not snippet:
					continue
				web_search_docs.append({
					"title": title,
					"url": url,
					"snippet": snippet
				})
		except Exception as e:
			self.logger.error(f"JSON反序列化失败:{e}")
		
		return web_search_docs
	
	def validate_inputs(self, state: QueryGraphState):
		item_names = state.get("item_names")
		rewritten_query = state.get("rewritten_query")
		
		if not item_names or not isinstance(item_names, list):
			raise ValidationError(node_name=self.name, message=f"输入参数 [item_names] 校验失败")
		
		if not rewritten_query or not isinstance(rewritten_query, str):
			raise ValidationError(node_name=self.name, message=f"输入参数 [rewritten_query] 校验失败")
		
		return item_names, rewritten_query


if __name__ == '__main__':
	setup_logging()
	print("开始测试查询节点-MCP网络查询节点")
	_state = {
		"rewritten_query": "今天科技圈有哪些大事？",
		"item_names": ["H3C LA2608 室内无线网关"]
	}
	webSearchMCPNode = WebSearchMCPNode()
	state = webSearchMCPNode.process(_state)
	print(state)
