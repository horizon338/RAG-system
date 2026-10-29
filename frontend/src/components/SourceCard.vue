<template>
  <div v-if="sources.length" class="sources">
    <div class="label">
      <el-icon><CollectionTag /></el-icon>
      引用来源 <span class="count">({{ sources.length }})</span>
    </div>
    <el-collapse v-model="activeNames" class="collapse">
      <el-collapse-item
        v-for="(s, i) in sources"
        :key="i"
        :name="i"
      >
        <template #title>
          <div class="title-row">
            <span class="src-name">📄 {{ s.source }}</span>
            <el-tag v-if="s.page" size="small" type="info">p.{{ s.page }}</el-tag>
            <span class="score">相似度 {{ (s.score * 100).toFixed(1) }}%</span>
          </div>
        </template>
        <div class="preview">{{ s.preview }}...</div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  sources: { type: Array, default: () => [] },
})

const activeNames = ref([0]) // 默认展开第一个

watch(
  () => props.sources.length,
  () => { activeNames.value = [0] }
)
</script>

<style scoped>
.sources {
  margin-bottom: 12px;
  background: #f8fafc;
  border-radius: 8px;
  padding: 8px 12px;
}
.label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  font-weight: 600;
  color: #475569;
  margin-bottom: 6px;
}
.count {
  color: #94a3b8;
  font-weight: 400;
}
.collapse {
  border: none;
  background: transparent;
}
.collapse :deep(.el-collapse-item__header) {
  background: transparent;
  border-bottom: none;
  font-size: 13px;
  height: auto;
  padding: 6px 0;
}
.collapse :deep(.el-collapse-item__content) {
  padding-bottom: 8px;
  font-size: 12px;
  color: #64748b;
  line-height: 1.6;
}
.title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}
.src-name {
  font-weight: 500;
  color: #1f2937;
}
.score {
  margin-left: auto;
  font-size: 12px;
  color: #64748b;
  background: #e0e7ff;
  color: #4f46e5;
  padding: 2px 8px;
  border-radius: 10px;
}
.preview {
  background: white;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
  color: #475569;
}
</style>
