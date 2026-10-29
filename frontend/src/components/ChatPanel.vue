<template>
  <el-card class="chat-card" shadow="never">
    <template #header>
      <div class="card-header">
        <span>💬 知识库问答</span>
        <el-tag size="small" v-if="streaming" type="primary" effect="plain">
          生成中...
          <span class="dot" />
        </el-tag>
      </div>
    </template>

    <!-- 对话列表 -->
    <div ref="listRef" class="chat-list">
      <!-- 欢迎提示 -->
      <div v-if="messages.length === 0" class="welcome">
        <el-icon :size="40" color="#4f7cff"><ChatDotRound /></el-icon>
        <h2>RAG 智能问答助手</h2>
        <p>先上传你的文档，然后就可以开始提问啦！</p>
        <div class="quick-questions">
          <el-button
            v-for="q in quickQs"
            :key="q"
            round
            size="small"
            plain
            @click="askQuestion(q)"
          >
            {{ q }}
          </el-button>
        </div>
      </div>

      <div
        v-for="(m, i) in messages"
        :key="i"
        class="msg"
        :class="m.role"
      >
        <el-avatar :size="32" :class="m.role + '-avatar'">
          {{ m.role === 'user' ? '我' : 'AI' }}
        </el-avatar>
        <div class="bubble">
          <!-- 用户消息直接显示 -->
          <div v-if="m.role === 'user'" class="text">{{ m.content }}</div>

          <!-- AI 消息: 引用来源 + 回答 -->
          <template v-else>
            <SourceCard v-if="m.sources && m.sources.length" :sources="m.sources" />
            <div class="answer-box">
              <div class="answer chat-markdown" v-html="renderMarkdown(m.content)" />
              <span v-if="m.streaming" class="cursor">▍</span>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="input-area">
      <el-input
        v-model="question"
        type="textarea"
        :rows="2"
        :disabled="streaming"
        placeholder="输入你的问题，回车发送（Shift+回车换行）"
        @keydown.enter.exact.prevent="askQuestion()"
        resize="none"
      />
      <div class="send-row">
        <el-tooltip content="清空对话">
          <el-button :icon="Delete" circle @click="clearChat" />
        </el-tooltip>
        <template v-if="streaming">
          <el-tooltip content="停止生成">
            <el-button type="danger" :icon="VideoPause" circle @click="stopStream" />
          </el-tooltip>
        </template>
        <template v-else>
          <el-tooltip content="发送">
            <el-button
              type="primary"
              :icon="Promotion"
              circle
              :disabled="!question.trim()"
              @click="askQuestion()"
            />
          </el-tooltip>
        </template>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, nextTick, onUnmounted } from 'vue'
import { marked } from 'marked'
import { ElMessage } from 'element-plus'
import { Delete, Promotion, VideoPause } from '@element-plus/icons-vue'
import { askStream, cancelStream } from '../api'
import SourceCard from './SourceCard.vue'

marked.setOptions({ breaks: true, gfm: true })

const props = defineProps({
  sources: { type: Array, default: () => [] },
})

const listRef = ref(null)
const question = ref('')
const messages = ref([])
const streaming = ref(false)
let controller = null

const quickQs = [
  '这个文档主要讲了什么？',
  '总结一下核心要点',
  '有哪些关键概念？',
]

const renderMarkdown = (text) => {
  if (!text) return '<span class="empty">...</span>'
  return marked.parse(text)
}

const scrollBottom = () => {
  nextTick(() => {
    if (listRef.value) {
      listRef.value.scrollTop = listRef.value.scrollHeight
    }
  })
}

const askQuestion = (preset) => {
  const q = preset ?? question.value
  if (!q.trim() || streaming.value) return

  // 追加用户消息
  messages.value.push({ role: 'user', content: q })
  question.value = ''
  streaming.value = true

  // 追加空的 AI 消息占位
  const aiIdx = messages.value.length
  messages.value.push({ role: 'ai', content: '', sources: [], streaming: true })
  scrollBottom()

  let sources = []
  controller = askStream(
    q,
    null,
    // onChunk
    (delta) => {
      messages.value[aiIdx].content += delta
      scrollBottom()
    },
    // onSource
    (src) => {
      sources = src
      messages.value[aiIdx].sources = src
      window.dispatchEvent(new CustomEvent('rag-sources', { detail: src }))
      scrollBottom()
    },
    // onDone
    () => {
      messages.value[aiIdx].streaming = false
      streaming.value = false
      controller = null
      scrollBottom()
    },
    // onError
    (err) => {
      messages.value[aiIdx].content = `❌ ${err}`
      messages.value[aiIdx].streaming = false
      streaming.value = false
      controller = null
      ElMessage.error(String(err))
    }
  )
}

const stopStream = () => {
  if (controller) cancelStream(controller)
  const last = messages.value[messages.value.length - 1]
  if (last && last.role === 'ai') {
    last.streaming = false
  }
  streaming.value = false
}

const clearChat = () => {
  if (streaming.value) stopStream()
  messages.value = []
}

onUnmounted(() => {
  if (controller) cancelStream(controller)
})
</script>

<style scoped>
.chat-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  border-radius: 12px;
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 600;
  font-size: 15px;
}
.dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  background: #4f7cff;
  border-radius: 50%;
  margin-left: 4px;
  animation: pulse 1s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}

.chat-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px 4px;
  min-height: 400px;
}

.welcome {
  text-align: center;
  padding: 40px 20px;
  color: #64748b;
}
.welcome h2 {
  margin: 12px 0 6px;
  color: #1f2937;
}
.welcome p {
  margin: 0 0 20px;
  font-size: 14px;
}
.quick-questions {
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
}

.msg {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
  align-items: flex-start;
}
.msg.user {
  flex-direction: row-reverse;
}
.bubble {
  max-width: 78%;
}
.user .bubble {
  background: #4f7cff;
  color: white;
  padding: 10px 14px;
  border-radius: 12px 12px 2px 12px;
}
.ai .bubble {
  background: #ffffff;
  padding: 10px 14px;
  border-radius: 12px 12px 12px 2px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.user-avatar { background: #4f7cff; color: white; font-size: 13px; }
.ai-avatar { background: linear-gradient(135deg, #6366f1, #4f7cff); color: white; font-size: 13px; }

.answer-box {
  margin-top: 8px;
}
.cursor {
  display: inline-block;
  color: #4f7cff;
  animation: blink 0.8s infinite;
}
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.input-area {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding-top: 12px;
  border-top: 1px solid #f1f5f9;
}
.send-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
