<template>
  <el-card class="doc-card" shadow="never">
    <template #header>
      <div class="card-header">
        <span>📚 知识库管理</span>
        <el-tag size="small" effect="light">
          {{ stats.documents }} 文档 / {{ stats.chunks }} 分块
        </el-tag>
      </div>
    </template>

    <el-upload
      class="uploader"
      drag
      multiple
      :auto-upload="false"
      :on-change="onChange"
      :show-file-list="true"
      file-list-type="text"
      :file-list="fileList"
      :accept="'.pdf,.txt,.md,.docx'"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        将文件拖到此处，或 <em>点击选择文件</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">
          支持 PDF / TXT / Markdown / DOCX
        </div>
      </template>
    </el-upload>

    <div class="actions">
      <el-button
        type="primary"
        :loading="uploading"
        :disabled="fileList.length === 0"
        @click="doUpload"
      >
        <el-icon><CaretRight /></el-icon>
        {{ uploading ? '正在入库...' : '开始入库' }}
      </el-button>
      <el-button
        :loading="rebuilding"
        :disabled="!stats.documents"
        @click="doRebuild"
      >
        重建索引
      </el-button>
      <el-button
        type="danger"
        plain
        :disabled="!stats.documents"
        @click="doReset"
      >
        清空知识库
      </el-button>
    </div>

    <!-- 已有来源列表 -->
    <div v-if="stats.sources && stats.sources.length" class="sources-list">
      <div class="label">📂 已入库来源</div>
      <el-scrollbar height="180px">
        <div
          v-for="s in stats.sources"
          :key="s"
          class="source-item"
        >
          <el-icon><Document /></el-icon>
          <span>{{ s }}</span>
        </div>
      </el-scrollbar>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { uploadFiles, rebuildIndex, resetKnowledge } from '../api'

const props = defineProps({
  stats: { type: Object, default: () => ({ documents: 0, chunks: 0, ready: false, sources: [] }) },
})
const emit = defineEmits(['refresh', 'after-upload'])

const uploadFilesLocal = ref([]) // 原始 File 对象
const uploading = ref(false)
const rebuilding = ref(false)

const fileList = computed(() =>
  uploadFilesLocal.value.map((f, i) => ({
    name: f.name,
    uid: i,
    status: 'ready',
  }))
)

const onChange = (file) => {
  // 覆盖整个列表为当前选择
  uploadFilesLocal.value = [file.raw]
}

const doUpload = async () => {
  if (!uploadFilesLocal.value.length) return
  uploading.value = true
  try {
    const { data } = await uploadFiles(uploadFilesLocal.value)
    ElMessage.success(`成功入库 ${data.chunks_added} 个分块，共 ${data.total_chunks}`)
    uploadFilesLocal.value = []
    emit('after-upload')
  } catch (e) {
    ElMessage.error(`上传失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    uploading.value = false
  }
}

const doRebuild = async () => {
  try {
    await ElMessageBox.confirm('将基于已上传文件重建索引，继续？', '确认')
    rebuilding.value = true
    const { data } = await rebuildIndex()
    ElMessage.success(data.message || '重建完成')
    emit('refresh')
  } catch {
    /* 取消 */
  } finally {
    rebuilding.value = false
  }
}

const doReset = async () => {
  try {
    await ElMessageBox.confirm(
      '将清空知识库并删除所有上传文件，此操作不可恢复！',
      '⚠️ 危险操作',
      { confirmButtonText: '确认清空', type: 'error' }
    )
    await resetKnowledge()
    uploadFilesLocal.value = []
    ElMessage.success('知识库已清空')
    emit('refresh')
  } catch {
    /* 取消 */
  }
}
</script>

<style scoped>
.doc-card {
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
.uploader {
  margin-bottom: 16px;
}
.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.sources-list {
  border-top: 1px solid #f1f5f9;
  padding-top: 12px;
}
.label {
  font-size: 13px;
  font-weight: 500;
  color: #64748b;
  margin-bottom: 6px;
}
.source-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  font-size: 13px;
  color: #374151;
  border-radius: 6px;
}
.source-item:hover {
  background: #f1f5f9;
}
.el-icon--upload {
  font-size: 38px;
  color: #4f7cff;
}
</style>
