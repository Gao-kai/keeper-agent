import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import ChatPage from './components/ChatPage'
import UploadPage from './components/UploadPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 默认进入对话页 */}
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/upload" element={<UploadPage />} />
        {/* 未匹配路径兜底到对话页 */}
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
