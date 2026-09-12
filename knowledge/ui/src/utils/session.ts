/** localStorage 中保存会话 ID 的键名 */
const SESSION_KEY = 'shopkeeper_session_id'

/** 生成一个 UUID v4 作为会话 ID */
function createSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  // 兜底：非安全上下文下 crypto.randomUUID 不可用
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/** 读取本地会话 ID，不存在时生成并持久化 */
export function getOrCreateSessionId(): string {
  let sessionId = localStorage.getItem(SESSION_KEY)
  if (!sessionId) {
    sessionId = createSessionId()
    localStorage.setItem(SESSION_KEY, sessionId)
  }
  return sessionId
}

/** 清空会话时移除本地会话 ID（下次进入将开启新会话） */
export function removeSessionId(): void {
  localStorage.removeItem(SESSION_KEY)
}
