<template>
  <el-container class="app">
    <!-- 顶部栏 -->
    <el-header class="app-header">
      <div class="brand">
        <el-icon :size="26" color="#4f7cff"><Cpu /></el-icon>
        <span class="title">RAG 智能问答系统</span>
        <el-tag v-if="stats.ready" type="success" effect="light" round>知识库就绪</el-tag>
        <el-tag v-else type="warning" effect="light" round>请先上传文档</el-tag>
      </div>
      <div class="stats">
        <el-tooltip content="文档数量">
          <el-badge :value="stats.documents" :max="999">
            <el-icon :size="20"><Collection /></el-icon>
          </el-badge>
        </el-tooltip>
        <el-tooltip content="分块数量">
          <el-badge :value="stats.chunks" :max="9999">
            <el-icon :size="20"><Document /></el-icon>
          </el-badge>
        </el-tooltip>
      </div>
    </el-header>

    <el-main class="app-main">
      <el-row :gutter="20">
        <!-- 左侧: 知识库管理 -->
        <el-col :span="8" class="side-col">
          <DocUpload
            :stats="stats"
            :source-chunks="sourceChunks"
            @refresh="fetchStats"
            @after-upload="fetchStats"
          />
        </el-col>

        <!-- 右侧: 对话 -->
        <el-col :span="16" class="chat-col">
          <ChatPanel />
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import DocUpload from './components/DocUpload.vue'
import ChatPanel from './components/ChatPanel.vue'
import { getStats } from './api'

const stats = ref({ documents: 0, chunks: 0, ready: false, sources: [] })
const sourceChunks = ref({})

let pollTimer = null

const fetchStats = async () => {
  try {
    const { data } = await getStats()
    stats.value = data.stats
    sourceChunks.value = data.source_chunks || {}
  } catch (e) {
    // 后端未启动时静默
  }
}

onMounted(() => {
  fetchStats()
  // 每 10 秒轮询一次状态（用户清空/重建后自动刷新）
  pollTimer = setInterval(fetchStats, 10000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.app {
  height: 100vh;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid #e5e7eb;
  padding: 0 24px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.title {
  font-size: 20px;
  font-weight: 600;
  color: #1f2937;
}
.stats {
  display: flex;
  gap: 18px;
  align-items: center;
}
.app-main {
  padding: 20px;
  height: calc(100vh - 60px);
  overflow: hidden;
}
/* 让 el-row / el-col 也参与高度传递，否则里面的 el-card 拿不到 height:100% */
.app-main :deep(.el-row) {
  height: 100%;
  display: flex;
}
.app-main :deep(.el-col) {
  height: 100%;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.side-col {
  height: 100%;
}
.chat-col {
  height: 100%;
}
</style>
