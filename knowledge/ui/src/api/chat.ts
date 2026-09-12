/** 开发环境走 /api 代理（vite 转发到本地 FastAPI），生产环境与后端同源直连 */
const API_BASE = import.meta.env.DEV ? '/api' : ''

export interface QueryRequest {
  query: string
  session_id: string
  is_stream: boolean
}

export interface StreamSubmitResponse {
  message: string
  session_id: string
  task_id: string
}

export interface QueryResponse {
  message: string
  session_id: string
  answer: string
}

export interface HistoryItem {
  /** 后端实际返回 _id，兼容 id */
  id?: string
  _id?: string
  role: 'user' | 'assistant' | string
  text: string
  /** 后端实际返回 timestamp（秒级），兼容 ts */
  ts?: number | null
  timestamp?: number | null
}

export interface HistoryResponse {
  session_id: string
  items: HistoryItem[]
}

/** SSE progress 事件携带的节点进度信息 */
export interface StreamProgress {
  status: string
  done_list: string[]
  running_list: string[]
}

export interface StreamHandlers {
  onReady?: () => void
  onProgress?: (p: StreamProgress) => void
  onDelta?: (text: string) => void
  onFinal?: (answer: string) => void
  onError?: (err: Error) => void
  /** 流意外结束（未收到 final 就断开）时回调 */
  onClose?: () => void
}

/** 提交查询任务；is_stream=true 时返回 task_id 供建立 SSE 连接 */
export async function submitQuery(req: QueryRequest): Promise<QueryResponse | StreamSubmitResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw new Error(`提交查询失败（HTTP ${res.status}）`)
  }
  return res.json()
}

/**
 * 建立 SSE 连接接收流式输出。
 * 使用 fetch + ReadableStream 手动解析（而非 EventSource）：
 * 后端 sse_generator 是无限循环、依赖客户端断开来结束，EventSource 的自动重连
 * 会造成重复连接；fetch 方式可在收到 final 后主动 abort，精确控制生命周期。
 *
 * @returns abort 函数，用于主动断开连接
 */
export function connectStream(taskId: string, handlers: StreamHandlers): () => void {
  const controller = new AbortController()
  let closed = false

  const finish = (fn?: (err: Error) => void, err?: Error) => {
    if (closed) return
    closed = true
    controller.abort()
    fn?.(err as Error)
  }

  ;(async () => {
    try {
      const res = await fetch(`${API_BASE}/stream/${encodeURIComponent(taskId)}`, {
        signal: controller.signal,
        headers: { Accept: 'text/event-stream' },
      })
      if (!res.ok || !res.body) {
        throw new Error(`建立 SSE 连接失败（HTTP ${res.status}）`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE 报文以空行分隔，逐块解析
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() ?? ''

        for (const block of blocks) {
          let event = 'message'
          const dataLines: string[] = []
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) {
              event = line.slice('event:'.length).trim()
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice('data:'.length).trimStart())
            }
          }
          if (!dataLines.length) continue

          let data: Record<string, unknown>
          try {
            data = JSON.parse(dataLines.join('\n'))
          } catch {
            console.error('SSE data 解析失败:', dataLines.join('\n'))
            continue
          }

          switch (event) {
            case 'ready':
              handlers.onReady?.()
              break
            case 'progress':
              handlers.onProgress?.({
                status: String(data.status ?? ''),
                done_list: (data.done_list as string[]) ?? [],
                running_list: (data.running_list as string[]) ?? [],
              })
              break
            case 'delta':
              handlers.onDelta?.(String(data.delta ?? ''))
              break
            case 'final':
              handlers.onFinal?.(String(data.answer ?? ''))
              finish()
              return
            default:
              break
          }
        }
      }
      // 服务端结束但未收到 final
      finish(handlers.onClose)
    } catch (err) {
      if (controller.signal.aborted) return // 主动 abort，不算错误
      finish(handlers.onError, err instanceof Error ? err : new Error('SSE 连接异常'))
    }
  })()

  return () => {
    finish(handlers.onClose)
  }
}

/** 读取指定会话的历史消息 */
export async function getHistory(sessionId: string, limit = 50): Promise<HistoryResponse> {
  const res = await fetch(`${API_BASE}/history/${encodeURIComponent(sessionId)}?limit=${limit}`)
  if (!res.ok) {
    throw new Error(`读取历史会话失败（HTTP ${res.status}）`)
  }
  return res.json()
}

/** 清空指定会话的历史记录 */
export async function clearHistory(sessionId: string): Promise<number> {
  const res = await fetch(`${API_BASE}/history/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    throw new Error(`清空会话失败（HTTP ${res.status}）`)
  }
  const data = (await res.json()) as { deleted_count?: number }
  return data.deleted_count ?? 0
}
