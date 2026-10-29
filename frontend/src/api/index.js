import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// 文档
export const uploadFiles = (files) => {
  const fd = new FormData()
  files.forEach((f) => fd.append('files', f))
  return http.post('/documents/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const getStats = () => http.get('/documents/stats')
export const rebuildIndex = () => http.post('/documents/rebuild')
export const resetKnowledge = () => http.delete('/documents/reset')

// 问答
export const ask = (question, topK) =>
  http.post('/chat/ask', { question, top_k: topK })

export const askStream = (question, topK, onChunk, onSource, onDone, onError) => {
  const controller = new AbortController()
  fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        // SSE 按 \n\n 分割
        let idx
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          const line = raw.replace(/^data:\s*/, '').trim()
          if (!line || line === '[DONE]') continue
          try {
            const chunk = JSON.parse(line)
            if (chunk.type === 'sources') onSource && onSource(chunk.sources)
            else if (chunk.type === 'token') onChunk && onChunk(chunk.content)
            else if (chunk.type === 'done') onDone && onDone()
            else if (chunk.type === 'error') onError && onError(chunk.content)
          } catch (e) {
            // 忽略解析失败
          }
        }
      }
      onDone && onDone()
    })
    .catch((e) => {
      if (e.name !== 'AbortError') onError && onError(String(e))
    })

  return controller // 用于中断
}

export const cancelStream = (controller) => controller && controller.abort()
