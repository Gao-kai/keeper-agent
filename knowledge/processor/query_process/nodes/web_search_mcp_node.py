"""
AI Agent是干嘛的（代理程序员干活的）
HTTP调用查询天气这个事情，只有两个路：
1. 程序员自己手动编码 自己调用axios.get
2. 程序员让Agent代替自己自动干 先创建agent 然后agent.invoke

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