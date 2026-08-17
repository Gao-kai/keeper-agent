import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  File,
  FileArchive,
  FileImage,
  FileSpreadsheet,
  FileText,
  Loader2,
  Trash2,
  Upload,
  UploadCloud,
} from 'lucide-react'
import {
  getTaskStatus,
  uploadFile,
  type TaskStatus,
  type UploadResult,
} from '../api/upload'

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

/** 轮询间隔（毫秒） */
const POLL_INTERVAL = 2000
/** 终止状态：命中后停止轮询 */
const TERMINAL_STATUS = ['completed', 'failed']

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function fileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg'].includes(ext)) return FileImage
  if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) return FileArchive
  if (['xls', 'xlsx', 'csv'].includes(ext)) return FileSpreadsheet
  if (['pdf', 'doc', 'docx', 'txt', 'md', 'ppt', 'pptx'].includes(ext)) return FileText
  return File
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [status, setStatus] = useState<UploadStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState('')
  const [tip, setTip] = useState('')
  const [taskInfo, setTaskInfo] = useState<TaskStatus | null>(null)
  const dragCounter = useRef(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const uploading = status === 'uploading'

  // 上传成功后每 2 秒轮询任务状态；命中终止状态后停止
  useEffect(() => {
    const taskId = result?.task_id
    if (status !== 'success' || !taskId) return

    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      if (cancelled) return
      try {
        const info = await getTaskStatus(taskId)
        if (cancelled) return
        setTaskInfo(info)
        // completed / failed 停止轮询；processing（或空）继续
        if (!TERMINAL_STATUS.includes(info.status)) {
          timer = window.setTimeout(poll, POLL_INTERVAL)
        }
      } catch {
        // 网络抖动时继续重试
        if (!cancelled) {
          timer = window.setTimeout(poll, POLL_INTERVAL)
        }
      }
    }

    setTaskInfo(null)
    poll()

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [status, result?.task_id])

  const selectFile = (next: File) => {
    if (uploading) return
    setFile(next)
    setStatus('idle')
    setProgress(0)
    setResult(null)
    setTaskInfo(null)
    setError('')
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    dragCounter.current = 0
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (!files.length) return
    if (files.length > 1) {
      setTip('一次只能上传一个文件，已保留第一个文件')
    } else {
      setTip('')
    }
    selectFile(files[0])
  }

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    dragCounter.current += 1
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    dragCounter.current -= 1
    if (dragCounter.current <= 0) setIsDragging(false)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.files?.[0]
    if (next) {
      setTip('')
      selectFile(next)
    }
    e.target.value = '' // 清空 input 值，允许重复选择同一文件
  }

  const clearFile = () => {
    if (uploading) return
    setFile(null)
    setStatus('idle')
    setResult(null)
    setTaskInfo(null)
    setError('')
    setTip('')
  }

  const handleUpload = async () => {
    if (!file || uploading) return
    setStatus('uploading')
    setProgress(0)
    setResult(null)
    setTaskInfo(null)
    setError('')
    try {
      const res = await uploadFile(file, (p) => setProgress(p))
      setResult(res)
      setStatus('success')
    } catch (err) {
      setError(err instanceof Error ? err.message : '上传失败，请稍后重试')
      setStatus('error')
    }
  }

  const Icon = file ? fileIcon(file.name) : UploadCloud

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <main className="w-full max-w-xl">
        <header className="mb-6 text-center">
          <h1 className="text-2xl font-semibold tracking-wide">掌柜智库</h1>
          <p className="mt-1 text-sm text-muted">上传知识文件，构建专属 RAG 知识库</p>
        </header>

        <section className="rounded-lg border border-white/10 bg-surface p-6 shadow-xl shadow-black/30">
          {/* 拖拽 / 点击选择区域 */}
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={[
              'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-12 transition-all',
              isDragging
                ? 'border-primary bg-primary/10'
                : 'border-white/20 bg-surfaceLight/40 hover:border-primary/60 hover:bg-surfaceLight/70',
            ].join(' ')}
          >
            <UploadCloud
              className={`h-12 w-12 ${isDragging ? 'text-primary-light' : 'text-muted'}`}
            />
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">
                {isDragging ? '松开鼠标开始上传' : '点击选择或拖拽文件到此处'}
              </p>
              <p className="mt-1 text-xs text-muted">
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
            <div className="mt-4 flex items-center gap-3 rounded-lg border border-white/10 bg-surfaceLight/60 p-3">
              <Icon className="h-9 w-9 shrink-0 text-primary-light" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-foreground">{file.name}</p>
                <p className="text-xs text-muted">{formatSize(file.size)}</p>
              </div>
              {!uploading && (
                <button
                  onClick={clearFile}
                  aria-label="移除文件"
                  className="rounded-md p-1.5 text-muted transition-colors hover:bg-white/10 hover:text-foreground"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>
          )}

          {tip && <p className="mt-3 text-xs text-amber-400">{tip}</p>}

          {/* 上传按钮 */}
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className={[
              'mt-4 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors',
              !file || uploading
                ? 'cursor-not-allowed bg-white/10 text-muted'
                : 'bg-primary text-white hover:bg-primary-dark',
            ].join(' ')}
          >
            {uploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                {progress > 0 ? `上传中 ${progress}%` : '上传中...'}
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                开始上传
              </>
            )}
          </button>

          {/* 上传进度条 */}
          {uploading && (
            <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-200"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          {/* 上传结果 */}
          {status === 'success' && result && (
            <>
              <div className="mt-4 flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-400">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <p className="font-medium">{result.message}</p>
                  <p className="mt-0.5 break-all text-xs text-emerald-400/80">
                    任务 ID：{result.task_id}
                  </p>
                </div>
              </div>
              <TaskProgress taskId={result.task_id} info={taskInfo} />
            </>
          )}

          {status === 'error' && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-400">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <p>{error}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

/** 任务执行进度展示：轮询到的最新状态实时渲染 */
function TaskProgress({ taskId, info }: { taskId: string; info: TaskStatus | null }) {
  const isCompleted = info?.status === 'completed'
  const isFailed = info?.status === 'failed'
  const hasNodes = (info && (info.running_list.length > 0 || info.completed_list.length > 0)) ?? false

  const badgeClass = isCompleted
    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
    : isFailed
      ? 'border-rose-500/30 bg-rose-500/10 text-rose-400'
      : 'border-primary/30 bg-primary/10 text-primary-light'
  const badgeText = isCompleted ? '已完成' : isFailed ? '执行失败' : '处理中'

  return (
    <div className="mt-4 rounded-lg border border-white/10 bg-surfaceLight/40 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium text-foreground">任务执行进度</p>
        <div className="flex items-center gap-2">
          {!isCompleted && !isFailed && (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-light" />
          )}
          <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${badgeClass}`}>
            {badgeText}
          </span>
        </div>
      </div>
      <p className="mt-1 break-all text-xs text-muted">任务 ID：{taskId}</p>

      <div className="mt-3 space-y-3">
        {/* 正在执行的节点 */}
        {info && info.running_list.length > 0 && (
          <div>
            <p className="text-xs text-muted">正在执行</p>
            <ul className="mt-1.5 space-y-1.5">
              {info.running_list.map((name) => (
                <li
                  key={name}
                  className="flex items-center gap-2 rounded-md border border-primary/20 bg-primary/5 px-3 py-1.5 text-sm text-foreground"
                >
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary-light" />
                  {name}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 已完成的节点 */}
        {info && info.completed_list.length > 0 && (
          <div>
            <p className="text-xs text-muted">已完成</p>
            <ul className="mt-1.5 space-y-1.5">
              {info.completed_list.map((name) => (
                <li
                  key={name}
                  className="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm text-emerald-400"
                >
                  <CheckCircle2 className="h-4 w-4 shrink-0" />
                  {name}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 尚无节点信息 */}
        {!hasNodes && (
          <p className="text-xs text-muted">任务已提交，等待节点调度...</p>
        )}
      </div>
    </div>
  )
}
