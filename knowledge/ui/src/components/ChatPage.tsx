import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { BookOpen, Eraser, Send, UploadCloud } from "lucide-react";
import {
  clearHistory,
  connectStream,
  getHistory,
  submitQuery,
  type QueryResponse,
  type StreamProgress,
  type StreamSubmitResponse,
} from "../api/chat";
import { getOrCreateSessionId, removeSessionId } from "../utils/session";
import ChatMessage, { type Message } from "./ChatMessage";

/** 将消息按天分组，渲染「今天 / 日期」分隔条 */
function DayDivider({ ts }: { ts: number }) {
  const d = new Date(ts * 1000);
  const today = new Date();
  const isToday = d.toDateString() === today.toDateString();
  const label = isToday
    ? "今天"
    : `${d.getMonth() + 1}月${d.getDate()}日`;

  return (
    <div className="flex items-center justify-center py-2">
      <span className="rounded-full bg-foreground/5 px-4 py-1 text-sm text-muted">
        {label}
      </span>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStream, setIsStream] = useState(true);
  const [sending, setSending] = useState(false);
  const [banner, setBanner] = useState("");

  const sessionIdRef = useRef(getOrCreateSessionId());
  const abortRef = useRef<(() => void) | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /** 更新指定消息（函数式，避免闭包旧值） */
  const patchMessage = useCallback(
    (id: string, patch: Partial<Message> | ((m: Message) => Partial<Message>)) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)),
      );
    },
    [],
  );

  // 首次进入：拉取历史会话还原聊天记录
  useEffect(() => {
    getHistory(sessionIdRef.current)
      .then((data) => {
        setMessages(
          data.items.map((item, i) => ({
            id: item._id ?? item.id ?? `h-${i}`,
            role: item.role === "user" ? "user" : "assistant",
            text: item.text,
            ts: item.ts ?? item.timestamp ?? Date.now() / 1000,
          })),
        );
      })
      .catch((err) => {
        // 无历史或网络异常时静默开始新会话
        console.error("恢复历史会话失败:", err);
      });
  }, []);

  // 新消息 / 流式增量到达时滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 组件卸载时断开未完成的 SSE 连接
  useEffect(() => {
    return () => abortRef.current?.();
  }, []);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || sending) return;

    setInput("");
    setBanner("");
    setSending(true);
    autoResize();

    const now = Date.now() / 1000;
    const assistantId = `a-${now}-${Math.random().toString(36).slice(2, 8)}`;

    setMessages((prev) => [
      ...prev,
      { id: `u-${now}`, role: "user", text, ts: now },
      { id: assistantId, role: "assistant", text: "", ts: now, streaming: true },
    ]);

    const fail = (err: unknown) => {
      console.error("查询失败:", err);
      patchMessage(assistantId, {
        streaming: false,
        error: err instanceof Error ? err.message : "查询失败，请稍后重试",
      });
      setSending(false);
    };

    if (isStream) {
      submitQuery({ query: text, session_id: sessionIdRef.current, is_stream: true })
        .then((res) => {
          const { task_id: taskId } = res as StreamSubmitResponse;
          abortRef.current = connectStream(taskId, {
            onProgress: (p: StreamProgress) =>
              patchMessage(assistantId, (m) => ({
                progress: { ...p, open: m.progress?.open ?? false },
              })),
            onDelta: (delta) => patchMessage(assistantId, (m) => ({ text: m.text + delta })),
            onFinal: (answer) => {
              patchMessage(assistantId, { text: answer, streaming: false, progress: undefined });
              setSending(false);
            },
            onError: fail,
            onClose: () => {
              // 未收到 final 就断开：仅结束 loading，保留已收到的增量文本
              patchMessage(assistantId, (m) => (m.streaming ? { streaming: false } : {}));
              setSending(false);
            },
          });
        })
        .catch(fail);
    } else {
      submitQuery({ query: text, session_id: sessionIdRef.current, is_stream: false })
        .then((res) => {
          patchMessage(assistantId, {
            text: (res as QueryResponse).answer,
            streaming: false,
          });
          setSending(false);
        })
        .catch(fail);
    }
  }, [input, sending, isStream, patchMessage]);

  /** 清空会话：断开进行中的流、清后端历史并重置本地会话 */
  const handleClear = useCallback(() => {
    if (sending) return;
    setBanner("");
    clearHistory(sessionIdRef.current)
      .then((count) => {
        console.log(`已清空 ${count} 条历史记录`);
        abortRef.current?.();
        abortRef.current = null;
        removeSessionId();
        sessionIdRef.current = getOrCreateSessionId();
        setMessages([]);
      })
      .catch((err) => {
        console.error("清空会话失败:", err);
        setBanner(err instanceof Error ? err.message : "清空会话失败，请稍后重试");
      });
  }, [sending]);

  /** 输入框高度自适应 */
  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 消息按天分组渲染（相邻消息同天只渲染一次分隔条）
  const rendered: React.ReactNode[] = [];
  let lastDay = "";
  messages.forEach((m) => {
    const day = new Date(m.ts * 1000).toDateString();
    if (day !== lastDay) {
      rendered.push(<DayDivider key={`d-${day}`} ts={m.ts} />);
      lastDay = day;
    }
    rendered.push(<ChatMessage key={m.id} message={m} />);
  });

  return (
    <div className="flex min-h-screen flex-col">
      {/* 顶栏（固定） */}
      <header className="fixed inset-x-0 top-0 z-10 border-b border-[#b3ada7]/50 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 lg:px-6">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-white shadow-sm">
            <BookOpen className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <h1 className="font-serif text-xl font-semibold leading-tight text-foreground">
              掌柜智库
            </h1>
            <p className="truncate text-xs text-muted">在线 · 可查询知识库与上传文档</p>
          </div>

          <div className="ml-auto flex items-center gap-2 sm:gap-4">
            {/* 流式输出开关 */}
            <label className="flex cursor-pointer select-none items-center gap-2 text-sm text-foreground">
              <input
                type="checkbox"
                checked={isStream}
                onChange={(e) => setIsStream(e.target.checked)}
                className="h-4 w-4 cursor-pointer accent-primary"
              />
              流式输出
            </label>

            <Link
              to="/upload"
              className="flex items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20"
            >
              <UploadCloud className="h-4 w-4" />
              <span className="hidden sm:inline">上传文档</span>
            </Link>

            <button
              onClick={handleClear}
              disabled={sending}
              className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Eraser className="h-4 w-4" />
              清空会话
            </button>
          </div>
        </div>
      </header>

      {/* 消息流区域 */}
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 pb-44 pt-24 lg:px-6">
        {messages.length === 0 ? (
          <div className="flex items-end gap-2.5">
            <div className="flex h-9 w-9 shrink-0 select-none items-center justify-center rounded-full bg-primary-light text-sm font-semibold text-white shadow-sm">
              掌
            </div>
            <div className="max-w-[75%]">
              <div className="rounded-lg rounded-bl-sm border border-[#b3ada7]/50 bg-surface px-4 py-3 text-base leading-relaxed text-foreground shadow-sm">
                <p>你好，我是掌柜智库客服。你可以直接提问，我会在「阶段进度」里展示处理过程。</p>
                <p className="mt-2 text-sm text-muted">
                  提示：可以先{" "}
                  <Link to="/upload" className="text-primary underline underline-offset-2 hover:text-primary-dark">
                    上传知识文档
                  </Link>
                  ，再针对文档内容提问。
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {rendered}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* 底部输入区（固定） */}
      <footer className="fixed inset-x-0 bottom-0 z-10 border-t border-[#b3ada7]/50 bg-background/85 backdrop-blur">
        <div className="mx-auto max-w-5xl px-4 pb-3 pt-3 lg:px-6">
          {banner && (
            <p className="mb-2 rounded-lg border border-error/40 bg-error/10 px-3 py-2 text-sm text-error">
              {banner}
            </p>
          )}
          <div className="flex items-end gap-3 rounded-xl border border-primary/30 bg-surface p-3 shadow-sm focus-within:border-primary/60">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoResize();
              }}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder="输入你的问题..."
              className="max-h-40 flex-1 resize-none bg-transparent text-base text-foreground outline-none placeholder:text-muted/70"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              className="flex shrink-0 items-center gap-1.5 rounded-lg bg-primary px-5 py-2.5 text-base font-medium text-white transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:bg-foreground/10 disabled:text-muted"
            >
              <Send className="h-4 w-4" />
              发送
            </button>
          </div>
          <p className="mt-2 text-xs text-muted">
            快捷键：Enter 发送，Shift + Enter 换行
          </p>
        </div>
      </footer>
    </div>
  );
}
