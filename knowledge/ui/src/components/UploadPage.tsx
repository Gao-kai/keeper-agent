import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  File,
  FileArchive,
  FileImage,
  FileSpreadsheet,
  FileText,
  ListTodo,
  Loader2,
  Trash2,
  Upload,
  UploadCloud,
} from "lucide-react";
import {
  getTaskStatus,
  uploadFile,
  type TaskStatus,
  type UploadResult,
} from "../api/upload";

type UploadStatus = "idle" | "uploading" | "success" | "error";

/** 轮询间隔（毫秒） */
const POLL_INTERVAL = 2000;
/** 终止状态：命中后停止轮询 */
const TERMINAL_STATUS = ["completed", "failed"];

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"].includes(ext))
    return FileImage;
  if (["zip", "rar", "7z", "tar", "gz"].includes(ext)) return FileArchive;
  if (["xls", "xlsx", "csv"].includes(ext)) return FileSpreadsheet;
  if (["pdf", "doc", "docx", "txt", "md", "ppt", "pptx"].includes(ext))
    return FileText;
  return File;
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");
  const [tip, setTip] = useState("");
  const [taskInfo, setTaskInfo] = useState<TaskStatus | null>(null);
  const dragCounter = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploading = status === "uploading";

  // 上传成功后每 2 秒轮询任务状态；命中终止状态后停止
  useEffect(() => {
    const taskId = result?.task_id;
    if (status !== "success" || !taskId) return;

    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      if (cancelled) return;
      try {
        const info = await getTaskStatus(taskId);
        if (cancelled) return;
        setTaskInfo(info);
        // completed / failed 停止轮询；processing（或空）继续
        if (!TERMINAL_STATUS.includes(info.status)) {
          timer = window.setTimeout(poll, POLL_INTERVAL);
        }
      } catch {
        // 网络抖动时继续重试
        if (!cancelled) {
          timer = window.setTimeout(poll, POLL_INTERVAL);
        }
      }
    };

    setTaskInfo(null);
    poll();

    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [status, result?.task_id]);

  const selectFile = (next: File) => {
    if (uploading) return;
    setFile(next);
    setStatus("idle");
    setProgress(0);
    setResult(null);
    setTaskInfo(null);
    setError("");
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (!files.length) return;
    if (files.length > 1) {
      setTip("一次只能上传一个文件，已保留第一个文件");
    } else {
      setTip("");
    }
    selectFile(files[0]);
  };

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current += 1;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) setIsDragging(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.files?.[0];
    if (next) {
      setTip("");
      selectFile(next);
    }
    e.target.value = ""; // 清空 input 值，允许重复选择同一文件
  };

  const clearFile = () => {
    if (uploading) return;
    setFile(null);
    setStatus("idle");
    setResult(null);
    setTaskInfo(null);
    setError("");
    setTip("");
  };

  const handleUpload = async () => {
    if (!file || uploading) return;
    setStatus("uploading");
    setProgress(0);
    setResult(null);
    setTaskInfo(null);
    setError("");
    try {
      const res = await uploadFile(file, (p) => setProgress(p));
      setResult(res);
      setStatus("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败，请稍后重试");
      setStatus("error");
    }
  };

  const Icon = file ? fileIcon(file.name) : UploadCloud;

  const progressBadgeClass =
    taskInfo?.status === "completed"
      ? "border-success/40 bg-success/10 text-success"
      : taskInfo?.status === "failed"
        ? "border-error/40 bg-error/10 text-error"
        : "border-primary/30 bg-primary/10 text-primary";
  const progressBadgeText =
    taskInfo?.status === "completed"
      ? "已完成"
      : taskInfo?.status === "failed"
        ? "执行失败"
        : "处理中";

  return (
    <div className="flex min-h-screen items-center justify-center p-6 lg:p-10">
      <main className="w-full max-w-6xl">
        <header className="mb-10 text-center">
          <h1 className="font-serif text-5xl font-semibold tracking-wide text-foreground lg:text-6xl">
            掌柜智库
          </h1>
          <p className="mt-3 text-xl text-muted">
            上传知识文件，构建专属 RAG 知识库
          </p>
        </header>

        <div className="grid grid-cols-1 items-stretch gap-8 lg:grid-cols-2">
          {/* 左栏：上传区域 */}
          <section className="rounded-lg border border-[#b3ada7]/60 bg-surface p-8 shadow-[0_4px_22px_rgba(120,98,81,0.12)] lg:p-10">
            {/* 拖拽 / 点击选择区域 */}
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={[
                "flex cursor-pointer flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed px-8 py-16 transition-all",
                isDragging
                  ? "border-primary bg-primary/10"
                  : "border-[#8b847d]/40 bg-surfaceLight/60 hover:border-primary/60 hover:bg-surfaceLight",
              ].join(" ")}
            >
              <UploadCloud
                className={`h-16 w-16 ${isDragging ? "text-primary" : "text-muted"}`}
              />
              <div className="text-center">
                <p className="text-xl font-medium text-foreground">
                  {isDragging ? "松开鼠标开始上传" : "点击选择或拖拽文件到此处"}
                </p>
                <p className="mt-2 text-base text-muted">
                  支持单个文件，推荐 PDF / Word / Markdown
                </p>
              </div>
              <input
                ref={inputRef}
                type="file"
                className="hidden"
                onChange={handleInputChange}
              />
            </div>

            {/* 选中文件信息 */}
            {file && (
              <div className="mt-6 flex items-center gap-4 rounded-lg border border-[#b3ada7]/40 bg-surfaceLight/80 p-4">
                <Icon className="h-12 w-12 shrink-0 text-primary" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-lg font-medium text-foreground">
                    {file.name}
                  </p>
                  <p className="text-base text-muted">
                    {formatSize(file.size)}
                  </p>
                </div>
                {!uploading && (
                  <button
                    onClick={clearFile}
                    aria-label="移除文件"
                    className="rounded-md p-2 text-muted transition-colors hover:bg-black/5 hover:text-foreground"
                  >
                    <Trash2 className="h-6 w-6" />
                  </button>
                )}
              </div>
            )}

            {tip && <p className="mt-4 text-base text-warning">{tip}</p>}

            {/* 上传按钮 */}
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className={[
                "mt-6 flex w-full items-center justify-center gap-2.5 rounded-lg px-5 py-4 text-lg font-medium transition-colors",
                !file || uploading
                  ? "cursor-not-allowed bg-[#252f3d]/10 text-muted"
                  : "bg-primary text-white hover:bg-primary-dark",
              ].join(" ")}
            >
              {uploading ? (
                <>
                  <Loader2 className="h-6 w-6 animate-spin" />
                  {progress > 0 ? `上传中 ${progress}%` : "上传中..."}
                </>
              ) : (
                <>
                  <Upload className="h-6 w-6" />
                  开始上传
                </>
              )}
            </button>

            {/* 上传进度条 */}
            {uploading && (
              <div className="mt-5 h-2.5 w-full overflow-hidden rounded-full bg-[#252f3d]/10">
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}

            {/* 上传结果 */}
            {status === "success" && result && (
              <div className="mt-6 flex items-start gap-2.5 rounded-lg border border-success/40 bg-success/10 p-4 text-lg text-success">
                <CheckCircle2 className="mt-0.5 h-6 w-6 shrink-0" />
                <div>
                  <p className="font-medium">{result.message}</p>
                  <p className="mt-1 break-all text-base text-success/80">
                    任务 ID：{result.task_id}
                  </p>
                </div>
              </div>
            )}

            {status === "error" && (
              <div className="mt-6 flex items-center gap-2.5 rounded-lg border border-error/40 bg-error/10 p-4 text-lg text-error">
                <AlertCircle className="h-6 w-6 shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </section>

          {/* 右栏：任务执行进度区域 */}
          <section className="flex flex-col rounded-lg border border-[#b3ada7]/60 bg-surface p-8 shadow-[0_4px_22px_rgba(120,98,81,0.12)] lg:p-10">
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-serif text-2xl font-semibold text-foreground">
                任务执行进度
              </h2>
              {status === "success" && result && (
                <span
                  className={`rounded-full border px-4 py-1.5 text-base font-medium ${progressBadgeClass}`}
                >
                  {progressBadgeText}
                </span>
              )}
            </div>

            {status === "success" && result ? (
              <div className="mt-4 flex max-h-[60vh] min-h-0 flex-1 flex-col">
                <p className="shrink-0 break-all text-base text-muted">
                  任务 ID：{result.task_id}
                </p>
                <div className="mt-4 min-h-0 flex-1 overflow-y-auto pr-1">
                  <TaskProgress info={taskInfo} />
                </div>
              </div>
            ) : (
              <div className="mt-6 flex flex-1 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-[#8b847d]/40 bg-surfaceLight/60 px-8 py-20 text-center">
                <ListTodo className="h-14 w-14 text-muted" />
                <p className="text-xl text-foreground">暂无运行中的任务</p>
                <p className="text-base text-muted">
                  上传文件后，此处将实时展示每个节点的执行进度
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

/** 任务执行进度：轮询到的最新状态实时渲染 */
function TaskProgress({ info }: { info: TaskStatus | null }) {
  const hasNodes =
    (info &&
      (info.running_list.length > 0 || info.completed_list.length > 0)) ??
    false;

  return (
    <div className="mt-6 space-y-6">
      {/* 正在执行的节点 */}
      {info && info.running_list.length > 0 && (
        <div>
          <p className="text-base text-muted">正在执行</p>
          <ul className="mt-3 space-y-2.5">
            {info.running_list.map((name) => (
              <li
                key={name}
                className="flex items-center gap-3 rounded-md border border-primary/20 bg-primary/5 px-4 py-3 text-lg text-foreground"
              >
                <Loader2 className="h-6 w-6 shrink-0 animate-spin text-primary" />
                {name}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 已完成的节点 */}
      {info && info.completed_list.length > 0 && (
        <div>
          <p className="text-base text-muted">已完成</p>
          <ul className="mt-3 space-y-2.5">
            {info.completed_list.map((name) => (
              <li
                key={name}
                className="flex items-center gap-3 rounded-md border border-success/30 bg-success/5 px-4 py-3 text-lg text-success"
              >
                <CheckCircle2 className="h-6 w-6 shrink-0" />
                {name}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 尚无节点信息 */}
      {!hasNodes && (
        <p className="text-lg text-muted">任务已提交，等待节点调度...</p>
      )}
    </div>
  );
}
