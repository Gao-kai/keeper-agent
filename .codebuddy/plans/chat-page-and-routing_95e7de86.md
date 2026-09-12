---
name: chat-page-and-routing
overview: 为「掌柜智库」前端新增 /chat 对话页面（含 SSE 流式问答、阶段进度、清空会话、历史恢复），引入路由使 /chat 为默认页、上传页迁移至 /upload，并完成与后端 query 相关接口的联调。
design:
  architecture:
    framework: react
  styleKeywords:
    - 暖色纸张网格
    - 简洁卡片化
    - 全站风格统一
  fontSystem:
    fontFamily: PingFang SC
    heading:
      size: 28px
      weight: 600
    subheading:
      size: 18px
      weight: 600
    body:
      size: 16px
      weight: 400
  colorSystem:
    primary:
      - "#4B607C"
      - "#6A9FCC"
      - "#394352"
    background:
      - "#EBE7E4"
      - "#F4F2F0"
      - "#EEF1F3"
    text:
      - "#252F3D"
      - "#5C5752"
    functional:
      - "#5DB87A"
      - "#E8993A"
      - "#E8704F"
todos:
  - id: setup-router
    content: 安装 react-router-dom 并改造 App.tsx 为路由出口（/ 重定向 /chat，/upload 指向现有上传页）
    status: completed
  - id: chat-api
    content: 新建 src/api/chat.ts 与 src/utils/session.ts，封装 query 提交、fetch 版 SSE 解析、历史读取/清空接口
    status: completed
  - id: chat-page-ui
    content: 新建 ChatPage.tsx，按截图布局实现顶栏（流式开关/清空会话）、消息流（气泡/头像/时间戳/日期分隔）、输入区（Enter 发送）
    status: completed
  - id: chat-stream-logic
    content: 实现流式/非流式发送逻辑、阶段进度折叠面板、delta 增量渲染与 final 收尾
    status: completed
  - id: chat-history
    content: 实现 session_id 持久化、刷新后 /history 会话恢复与清空会话联调
    status: completed
  - id: verify-e2e
    content: 启动 vite 与后端，验证 /chat 全流程（流式问答、非流式问答、清空、刷新恢复）
    status: completed
---

## Product Overview
在「掌柜智库」前端中新增用户对话页面，实现与 RAG 知识库的问答交互，并与后端 query 相关 API 完成联调。

## Core Features
- **路由改造**：新增路由体系，`/chat` 为对话页、`/upload` 承载现有上传页，访问 `/` 默认重定向到 `/chat`
- **对话界面**（布局参考截图，配色沿用现有暖色主题）：
  - 顶栏：知识库标题/在线状态、「流式输出」开关、「清空会话」按钮
  - 消息流：日期分隔（如"今天"）、欢迎提示气泡、用户/助手气泡（头像 + 时间戳），助手气泡内含可折叠「阶段进度」区（显示已完成/进行中的处理节点）
  - 底部输入区：多行输入框 + 发送按钮，Enter 发送、Shift+Enter 换行
- **API 联调**：
  - 流式模式：`POST /query (is_stream=true)` 拿到 `task_id` 后建立 SSE 连接（`GET /stream/{task_id}`），实时接收 `progress`（阶段进度）、`delta`（增量文本）、`final`（完整答案）事件
  - 非流式模式：`POST /query (is_stream=false)` 直接展示完整答案
  - 会话恢复：`session_id` 存 localStorage，刷新页面后通过 `GET /history/{session_id}` 还原聊天记录
  - 清空会话：`DELETE /history/{session_id}` 清空后端历史并重置本地会话


## Tech Stack
- 前端：React 18 + TypeScript + Vite 5（现有），新增依赖 `react-router-dom`
- 样式：Tailwind CSS 3，复用现有主题 token（`primary`/`surface`/`foreground`/`muted`/`success` 等）
- 后端：FastAPI（已有，本次不改后端代码）

## Implementation Approach
- **路由**：`App.tsx` 改为 `BrowserRouter` 路由出口；`/` 用 `<Navigate>` 重定向到 `/chat`。生产环境后端已通过 `mount("/")` 兜底返回 `index.html`，且已注册 `GET /chat`，前端路由刷新可直接命中
- **SSE 客户端**：使用 `fetch` + `ReadableStream` 手动解析 SSE 报文（而非 `EventSource`）。原因：后端 `sse_generator` 是无限循环、依赖客户端断开来结束，`EventSource` 的自动重连会造成重复连接；fetch 方式可精确控制生命周期（收到 `final` 后主动 abort）
- **会话管理**：首次进入生成 UUID 作为 `session_id` 存入 localStorage；已有 `session_id` 时调用 `/history` 还原消息列表
- **状态设计**：单条助手消息携带 `progress`（done_list/running_list）与流式 `answer`，折叠面板默认收起，`final` 到达后停止展示 loading

## Architecture Design
```mermaid
flowchart LR
    App["App.tsx (BrowserRouter)"] --> ChatPage["ChatPage /chat"]
    App --> UploadPage["UploadPage /upload"]
    ChatPage --> chatAPI["api/chat.ts"]
    chatAPI -->|POST /query| Backend["FastAPI 后端"]
    chatAPI -->|GET /stream/:task_id (SSE)| Backend
    chatAPI -->|GET·DELETE /history/:session_id| Backend
    ChatPage --> LS["localStorage (session_id)"]
```

## Directory Structure Summary
```
knowledge/ui/src/
├── App.tsx                    # [MODIFY] 改为 BrowserRouter 路由出口（/ 重定向 /chat、/chat、/upload）
├── api/
│   ├── upload.ts              # [不改] 现有上传 API
│   └── chat.ts                # [NEW] query 提交、SSE 流式连接（fetch 解析）、历史读取/清空
├── utils/
│   └── session.ts             # [NEW] session_id 的生成与 localStorage 读写
└── components/
    ├── UploadPage.tsx         # [不改] 迁移到 /upload 路由
    └── ChatPage.tsx           # [NEW] 对话页：顶栏(流式开关/清空会话)、消息流、阶段进度折叠面板、输入区
```

## Key Code Structures
```ts
// api/chat.ts 核心契约（与 query_schema.py / sse_push.py 对齐）
interface QueryRequest { query: string; session_id: string; is_stream: boolean }
interface StreamSubmitResponse { message: string; session_id: string; task_id: string }
interface QueryResponse { message: string; session_id: string; answer: string }
interface HistoryItem { id: string; role: "user" | "assistant"; text: string; ts: number | null }

// SSE 事件（GET /stream/{task_id}）
// ready → 通道就绪；progress → {status, done_list, running_list}
// delta → {delta}；final → {answer}（收到后关闭连接）

// 提交流式问答，返回 task_id
function submitQuery(req: QueryRequest): Promise<StreamSubmitResponse>
// 建立 SSE 连接，通过回调分发事件；返回 abort 函数用于主动断开
function connectStream(taskId: string, handlers: {
  onProgress?: (p: { status: string; done_list: string[]; running_list: string[] }) => void
  onDelta?: (text: string) => void
  onFinal?: (answer: string) => void
  onError?: (err: Error) => void
}): () => void
```

## Implementation Notes
- 复用 `api/upload.ts` 的 `API_BASE = import.meta.env.DEV ? '/api' : ''` 约定，`/api/stream/...` 经 vite 代理转发且代理支持流式透传
- 消息发送中禁止重复提交；组件卸载/清空会话时 abort 未完成的 SSE 连接
- 消息列表自动滚动到底部（新消息/delta 到达时）


## Design Style
沿用现有「掌柜智库」米色暖色纸张网格主题，布局按参考截图：顶部为标题栏卡片（衬线标题 + 在线状态副文案 + 流式输出开关 + 清空会话按钮）；中部消息流区域区分用户（右对齐、primary 色气泡）与助手（左对齐、surface 色气泡），气泡带圆形头像与时间戳，助手消息底部含可折叠「阶段进度」面板；底部为固定输入区卡片（多行输入框 + 发送按钮 + 快捷键提示）。交互上包含气泡入场动画、进度节点 spinner、发送按钮 hover 加深等微动效。
