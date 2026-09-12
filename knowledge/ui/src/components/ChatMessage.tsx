import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";
import type { StreamProgress } from "../api/chat";

export interface ChatProgress extends StreamProgress {
  /** 面板是否展开 */
  open: boolean;
}

/** 单条聊天消息的数据结构 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** 秒级时间戳；历史消息来自后端，本地消息为发送时刻 */
  ts: number;
  /** 处理阶段进度（仅助手消息，流式过程中更新） */
  progress?: ChatProgress;
  /** 是否仍在生成中（展示光标动画） */
  streaming?: boolean;
  /** 该条消息是否发送失败 */
  error?: string;
}

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

/** 处理中的节点数 */
export function runningCount(p?: ChatProgress): number {
  return p ? p.running_list.length : 0;
}

/** 已完成的节点数 */
export function doneCount(p?: ChatProgress): number {
  return p ? p.done_list.length : 0;
}

/** 助手消息内的「阶段进度」折叠面板 */
function ProgressPanel({ progress }: { progress: ChatProgress }) {
  const [open, setOpen] = useState(progress.open);
  const isFirstRender = useRef(true);

  // 流式过程中 open 值由父组件维护（初始折叠）
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    setOpen(progress.open);
  }, [progress.open]);

  const isDone = progress.status === "completed";
  const isFailed = progress.status === "failed";
  const summary = `已完成 ${doneCount(progress)}，进行中 ${runningCount(progress)}${
    progress.status ? `，状态：${isDone ? "已完成" : isFailed ? "失败" : "处理中"}` : ""
  }`;

  return (
    <div className="mt-3 rounded-md border border-dashed border-primary/30 bg-primary/5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm text-primary transition-colors hover:bg-primary/10"
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" />
        )}
        阶段进度
        <span className="ml-1 font-medium">{summary}</span>
        {!isDone && !isFailed && (
          <Loader2 className="ml-auto h-4 w-4 shrink-0 animate-spin text-primary" />
        )}
        {isDone && <CheckCircle2 className="ml-auto h-4 w-4 shrink-0 text-success" />}
        {isFailed && <AlertCircle className="ml-auto h-4 w-4 shrink-0 text-error" />}
      </button>

      {open && (
        <div className="space-y-4 px-4 pb-3 pt-1">
          {progress.running_list.length > 0 && (
            <div>
              <p className="text-xs text-muted">正在执行</p>
              <ul className="mt-1.5 space-y-1.5">
                {progress.running_list.map((name) => (
                  <li
                    key={name}
                    className="flex items-center gap-2 rounded border border-primary/20 bg-surfaceLight/80 px-3 py-1.5 text-sm text-foreground"
                  >
                    <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {progress.done_list.length > 0 && (
            <div>
              <p className="text-xs text-muted">已完成</p>
              <ul className="mt-1.5 space-y-1.5">
                {progress.done_list.map((name) => (
                  <li
                    key={name}
                    className="flex items-center gap-2 rounded border border-success/30 bg-success/5 px-3 py-1.5 text-sm text-success"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {progress.running_list.length === 0 && progress.done_list.length === 0 && (
            <p className="text-sm text-muted">任务已提交，等待节点调度...</p>
          )}
        </div>
      )}
    </div>
  );
}

interface ChatMessageProps {
  message: Message;
}

/** 单条消息气泡：用户右侧、助手左侧，带头像与时间戳 */
export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex animate-[message-in_0.25s_ease-out] items-end gap-2.5 ${
        isUser ? "flex-row-reverse" : ""
      }`}
    >
      {/* 头像 */}
      <div
        className={`flex h-9 w-9 shrink-0 select-none items-center justify-center rounded-full text-sm font-semibold text-white shadow-sm ${
          isUser ? "bg-primary" : "bg-primary-light"
        }`}
        title={isUser ? "我" : "掌柜智库"}
      >
        {isUser ? "我" : "掌"}
      </div>

      {/* 气泡 + 时间戳 */}
      <div className={`flex max-w-[75%] flex-col ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={[
            "rounded-lg px-4 py-3 text-base leading-relaxed shadow-sm",
            isUser
              ? "rounded-br-sm bg-primary text-white"
              : "rounded-bl-sm border border-[#b3ada7]/50 bg-surface text-foreground",
          ].join(" ")}
        >
          {message.error ? (
            <p className="flex items-center gap-2 text-error">
              <AlertCircle className="h-5 w-5 shrink-0" />
              {message.error}
            </p>
          ) : (
            <p className="whitespace-pre-wrap break-words">
              {message.text}
              {message.streaming && (
                <span className="ml-0.5 inline-block h-4 w-2 animate-pulse rounded-sm bg-primary/60 align-middle" />
              )}
            </p>
          )}

          {/* 阶段进度折叠面板 */}
          {message.progress && !isUser && (
            <ProgressPanel progress={message.progress} />
          )}
        </div>
        <span className="mt-1 px-1 text-xs text-muted">{formatTime(message.ts)}</span>
      </div>
    </div>
  );
}
