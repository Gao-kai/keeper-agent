export interface UploadResult {
  message: string
  task_id: string
}

export interface TaskStatus {
  status: string
  completed_list: string[]
  running_list: string[]
}

// 开发环境走 /api 代理（vite 转发到本地 FastAPI），生产环境与后端同源直连
const API_BASE = import.meta.env.DEV ? '/api' : ''

/** 查询任务当前执行状态 */
export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const res = await fetch(`${API_BASE}/status/${taskId}`)
  if (!res.ok) {
    throw new Error(`查询任务状态失败（HTTP ${res.status}）`)
  }
  return res.json() as Promise<TaskStatus>
}

/**
 * 上传单个文件到后端
 * 使用 XMLHttpRequest 以支持上传进度回调
 */
export function uploadFile(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/upload_file`)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResult)
        } catch {
          reject(new Error('响应解析失败'))
        }
      } else {
        reject(new Error(`上传失败（HTTP ${xhr.status}）`))
      }
    }

    xhr.onerror = () => reject(new Error('网络错误，请确认后端服务已启动'))
    xhr.send(formData)
  })
}
