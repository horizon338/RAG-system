# RAG 问答系统

基于 **检索增强生成（RAG）** 的本地知识库问答系统：上传文档 → 自动解析分块并向量化 → 提问时检索相关片段 → 由大模型基于片段生成带引用来源的回答。

> 原始设计文档见 `.trae/documents/rag_qa_system_plan.md`（实施前蓝图）。本文档反映实际落地状态。

---

## 技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| 后端 | Python 3.11+ / FastAPI / Uvicorn | 异步 Web 框架，自动生成 OpenAPI 文档 |
| 向量检索 | FAISS（IndexFlatIP，归一化后内积 = 余弦相似度） | CPU 版本，轻量单机 |
| 文本分块 | 自实现递归字符分块（中文标点优先切分） | 不依赖 LangChain |
| Embedding | 默认字符 n-gram 离线向量；可切换 BAAI/bge-small-zh-v1.5 | `embedding.py` 内置 Fallback，零模型依赖也能跑 |
| 大模型 | OpenAI 兼容接口 / DeepSeek / Ollama / Mock | HTTP 调用，无本地模型推理 |
| 文档解析 | pypdf / python-docx / 原生文本读取 | PDF / DOCX / TXT / Markdown |
| 前端 | Vue 3 + Vite + Element Plus + marked | SPA，流式打字机效果，多文件上传 |
| 数据存储 | 本地目录 `data/` | 上传文件 + FAISS 索引（serialize 绕过中文路径）+ 分块元数据（pickle） |

> **注意**：本项目不依赖 PyTorch、LangChain 训练/推理代码。Embedding 默认 Fallback 路径无任何三方模型下载；HF 路线（可选）启用时会间接拉起 `sentence-transformers` + torch。

---

## 目录结构

```
RAG问答系统/
├── backend/
│   ├── main.py                  # FastAPI 入口（路由 & 服务延迟注册）
│   ├── config.py                # Pydantic Settings，读取根目录 .env
│   ├── requirements.txt
│   ├── routers/
│   │   ├── documents.py         # 上传 / 重建 / 统计 / 删除单文档 / 清空
│   │   └── chat.py              # 同步问答 / SSE 流式问答（带 history）
│   └── services/
│       ├── document_loader.py   # PDF/TXT/MD/DOCX 解析 → Document 列表
│       ├── text_splitter.py     # 递归分块 + Document dataclass
│       ├── embedding.py         # Embedding 封装（HF 可切换，Fallback 默认）
│       ├── vector_store.py      # FAISS 持久化检索 + remove_by_source
│       ├── llm.py               # LLM 客户端（openai/deepseek/ollama/mock）
│       └── rag_pipeline.py      # 入库 / 检索 / Prompt 组装 / 多轮历史截断
├── frontend/
│   ├── src/
│   │   ├── App.vue              # 左右两栏布局 + 定时轮询 stats
│   │   ├── main.js
│   │   ├── api/index.js         # axios + fetch SSE
│   │   ├── components/
│   │   │   ├── DocUpload.vue    # 拖拽多文件上传 + 单文档删除 + 分块数展示
│   │   │   ├── ChatPanel.vue    # 对话列表（flex 三段式，中间滚动）
│   │   │   └── SourceCard.vue   # 引用来源折叠卡片
│   │   └── style.css            # Element Plus 主题覆盖 + markdown 样式
│   ├── vite.config.js           # /api 代理到 8000
│   └── package.json
├── data/                        # 运行时数据（自动创建）
│   ├── uploads/                 # 用户上传的原始文件
│   └── faiss_index/             # index.faiss + index.meta.pkl
├── test_documents/              # 示例文档
├── .env.example                 # 配置模板
├── .env                         # 本地配置（gitignore）
├── .gitignore
└── start.bat                    # 一键启动前后端
```

---

## 快速开始

1. 安装后端依赖：

```powershell
pip install -r backend/requirements.txt
```

2. （可选）安装前端依赖：

```powershell
cd frontend
npm install
```

3. 配置模型：复制 `.env.example` 为 `.env`，按需填写。最低配置下 `LLM_PROVIDER=mock` 也能演示完整流程。

4. 启动（两种方式任选）：

```powershell
# 方式 A：一键启动
start.bat

# 方式 B：手动
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

5. 打开 http://localhost:5173 ，上传文档后即可提问。API 文档：http://localhost:8000/docs 。

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查，返回当前 LLM 后端 |
| GET | `/api/documents/stats` | 知识库统计（文档数 / 分块数 / 来源 / `source_chunks` 按文档分块数） |
| POST | `/api/documents/upload` | 上传并入库（支持多文件） |
| POST | `/api/documents/rebuild` | 基于已上传文件重建索引 |
| DELETE | `/api/documents/remove/{filename}` | 删除单个文档（索引 + 原文件） |
| DELETE | `/api/documents/reset` | 清空知识库与所有上传文件 |
| POST | `/api/chat/ask` | 同步问答，支持 `history` 多轮历史 |
| POST | `/api/chat/stream` | SSE 流式问答（先推 sources，再推 token，最后 done） |

### 典型请求（多轮）

```json
POST /api/chat/ask
{
  "question": "它们难度如何？",
  "top_k": 4,
  "history": [
    { "role": "user",      "content": "这个文档有什么项目选题？" },
    { "role": "assistant", "content": "文档中有智慧寝室助理、智能咖啡机等选题..." }
  ]
}
```

---

## 实现要点

### 中文路径兼容
FAISS 的 `write_index/read_index` 在 Windows 中文路径下会报 Permission denied。改用 `faiss.serialize_index(index)` → Python 原生 `open().write()` 写入 bytes，读时 `np.frombuffer(bytes, dtype=np.uint8)` → `faiss.deserialize_index()` 还原，完全绕开 C++ 层路径。

### 离线降级链路
- **Embedding**：`TRY_HF = False` 硬编码，默认走 `_FallbackEmbedder`（384 维 n-gram 哈希向量）。若启用 HF，需取消 `requirements.txt` 中的 `sentence-transformers` 和 `langchain-huggingface`。
- **LLM**：配置缺失时自动降级 `MockLLM`，返回结构化占位回答。界面流程完整可演示。

### 启动速度
所有服务（vector_store / llm / rag_pipeline）与路由在 `lifespan` 阶段延迟 import，三方库导入不阻塞 ASGI 启动。

### 单文档删除
FAISS `IndexFlatIP` 无原生删除，`vector_store.remove_by_source()` 内部先过滤保留索引，再对剩余文档重新 Embedding + 重建索引 + 覆盖 save。

### 多轮对话历史
`rag_pipeline._build_messages()` 支持注入 `history`，并通过 `_truncate_history()` 自动截断到最近 6 条避免 token 爆炸。前端自动构造已完成的 user/assistant 对传给后端，流式中的 AI 回复不进入 history。

### 前端三段式滚动
`el-card` 内部有 `.el-card__body` 包装层，默认不参与 flex 高度分配。通过 `:deep(.el-card__body)` 强制 `flex:1; min-height:0; flex-direction:column; overflow:hidden`，让 `.chat-list` 独占中间剩余空间并 `overflow-y:auto`。`el-row` / `el-col` 同样通过 `min-height:0` 打通高度传递链。

### 流式容错
部分 OpenAI 中转站会在 SSE 流末尾返回 `choices` 为空、仅含 `usage` 的块，已做空值保护。

---

## 配置项（.env）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `openai` / `deepseek` / `ollama` / `mock` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | — | OpenAI 兼容接口（中转站 / 官方 / 国产模型） |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | — | DeepSeek 官方（可选） |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5:7b` | 本地 Ollama（可选） |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | HF Embedding（需网络 + 取消 requirements 注释） |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` |
| `CHUNK_SIZE` | `500` | 分块字符数 |
| `CHUNK_OVERLAP` | `50` | 分块重叠字符数 |
| `TOP_K` | `4` | 检索 Top-K |

---

## 与原始计划的差异

| 计划项 | 实际落地 | 原因 |
|---|---|---|
| LangChain 编排 | 自实现 `text_splitter` + `Document` | 零依赖更轻，避免三方版本锁死 |
| bge-m3 Embedding 强制加载 | 默认 FallbackEmbedder，HF 可切换 | 首次运行 2GB+ 下载影响启动 |
| 外部部署 / Docker | 无 | 单机演示目标 |
| 扫描 PDF OCR | 无 | 暂不涉及 |
| WebSocket 对话 | 采用 SSE `StreamingResponse` | 语义合适，实现更简单 |
| 自动触发 LLM API Key | `.env` 显式配置 | 安全优先 |

---

## 可优化发展方向（按优先级 / 工作量排序）

### 🟢 P0 — 低成本高收益

| 方向 | 说明 | 涉及文件 |
|---|---|---|
| **启用真实 Embedding 模型** | 取消 `requirements.txt` 里 `sentence-transformers` / `langchain-huggingface` 的注释，把 `embedding.py` 里 `TRY_HF = True`。Fallback n-gram 仅用于演示，真实检索质量差距很大 | `requirements.txt`, `backend/services/embedding.py` |
| **调整分块 / Top-K 参数** | 现在是经验值：`chunk_size=500`, `overlap=50`, `top_k=4`。用真实文档集评测后调整。可加到 stats 接口里暴露当前配置 | `backend/config.py`, `rag_pipeline.py` |
| **SSE 连接异常恢复** | 当前 `askStream` 被中断后只抛 error。前端可以记录当前 aiMsg 位置，点击"继续生成"时从断点重连 | `api/index.js`, `ChatPanel.vue` |
| **禁止相同文件名重复上传** | 现在 `DocUpload.vue` 用 uid 去重，但 `onChange` 累积逻辑在跨文件同名时仍可能重复。后端 `upload_files` 可以对重复文件名直接拒绝 | `routers/documents.py`, `DocUpload.vue` |

### 🟡 P1 — 中等工作量，效果明显

| 方向 | 说明 | 涉及文件 |
|---|---|---|
| **混合检索：向量 + BM25** | 纯向量对关键词精确匹配（如人名、编号）效果差。BM25 做关键词召回，向量做语义召回，合并后重排。Python 库可用 `rank_bm25` | 新模块 `backend/services/bm25_retriever.py`, `vector_store.py`, `rag_pipeline.py` |
| **Cross-Encoder 重排序（Rerank）** | 对 FAISS Top-K 结果用 bge-reranker-v2-m3 做精排，准确率显著提升。需要 HF，间接需要 torch | `rag_pipeline.py` 新增 rerank 步骤 |
| **查询改写（Query Rewrite）** | 用户的口语化问题（"它难不难？"）先让 LLM 改写为完整语义问题，再去检索。对多轮对话尤其有效 | `rag_pipeline.py` 新增 `rewrite_query()` |
| **异步上传 + 进度条** | 大 PDF 解析和向量化阻塞请求。改为 Celery/Redis 或 `asyncio` 后台任务，上传即返回 task_id，前端轮询进度 | `routers/documents.py`, 新增 Task 模块 |
| **按文档 / 知识集合隔离** | 给每个文档（或一组文档）一个 `collection_id`，检索时只在指定集合内搜。需要 FAISS 支持 Filtered Search（IndexIDMap） | `vector_store.py` 引入 `faiss.IndexIDMap`，加 collection 路由 |

### 🔵 P2 — 结构性升级

| 方向 | 说明 |
|---|---|
| **替换 FAISS 为更现代的向量库** | FAISS IndexFlatIP 精确但慢（线性扫描）。数据量变大时换 Chroma / Milvus / pgvector；或换 FAISS IVF/HNSW 近似索引 |
| **Embedding 和 Rerank 用 TorchScript 预编译** | 去掉线上 torch 动态依赖，部署体积小、启动快 |
| **扫描 PDF OCR** | 集成 Tesseract / PaddleOCR，扫描版 PDF 解析质量陡增 |
| **Agent 化 RAG（Plan-and-Answer / RAG-Fusion）** | 不只是"检索一段 → 回答"，而是让 LLM 自己决定：要不要检索、拆分成几个子问题、分别检索、汇总 |
| **评估体系** | 建立"问题-期望片段"标注集，自动评测召回率 / 答案质量。可用 RAGAS / TruLens |
| **多模态扩展** | 图片（OCR / CLIP 检索）、表格（Cohere Embed 专列）、视频帧 |
| **权限 / 多用户** | 每个用户独立知识库，JWT / Session 鉴权，支持共享与权限控制 |

### 🟣 P3 — 工程化 / 运维

| 方向 | 说明 |
|---|---|
| **Docker Compose 一键起** | 后端 + 前端 + Ollama 本地模型容器化 |
| **Prometheus + Grafana 监控** | API 延迟 / 检索耗时 / LLM 调用量可视化 |
| **API Key 配额管理** | 后端按用户限流、成本统计、超预算熔断 |
| **数据导出 / 备份** | 一键导出向量索引 + 元数据 + 原始文档，方便迁移 |
| **单元 / 集成测试** | 当前只有端到端手工验证，缺 pytest 用例 |

---

## 依赖一览（零 PyTorch 友好）

```
fastapi, uvicorn, python-multipart, pydantic, pydantic-settings
faiss-cpu, numpy
openai, httpx
pypdf, python-docx
python-dotenv

# Optional（HF Embedding 路线，启用时会间接引入 torch）
# sentence-transformers>=2.2.0
# langchain-huggingface>=0.3.0
```
