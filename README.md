# RAG 问答系统

基于 **检索增强生成（RAG）** 的本地知识库问答系统：上传文档 → 自动解析分块并向量化 → 提问时检索相关片段 → 由大模型基于片段生成带引用来源的回答。

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 | Python 3.11+ / FastAPI / Uvicorn |
| 向量检索 | FAISS（IndexFlatIP，归一化后内积 = 余弦相似度） |
| 文本分块 | 自实现递归字符分块（中文标点优先切分） |
| Embedding | 内置字符 n-gram 向量；网络可用时可切换 BAAI/bge-small-zh-v1.5 |
| 大模型 | OpenAI 兼容接口（中转站 / DeepSeek / 通义等）、Ollama、Mock 兜底 |
| 文档解析 | pypdf / python-docx / 原生文本读取 |
| 前端 | Vue 3 / Vite / Element Plus / marked |
| 数据存储 | 本地目录 `data/`（上传文件 + FAISS 索引 + 分块元数据） |

## 目录结构

```
RAG问答系统/
├── backend/
│   ├── main.py                  # FastAPI 入口（路由延迟注册）
│   ├── config.py                # 配置（读取根目录 .env）
│   ├── requirements.txt
│   ├── routers/
│   │   ├── documents.py         # 上传 / 重建 / 统计 / 清空
│   │   └── chat.py              # 同步问答 / SSE 流式问答
│   └── services/
│       ├── document_loader.py   # PDF/TXT/MD/DOCX 解析
│       ├── text_splitter.py     # 递归分块 + Document 结构
│       ├── embedding.py         # Embedding（含离线降级）
│       ├── vector_store.py      # FAISS 持久化检索
│       ├── llm.py               # LLM 客户端（openai/deepseek/ollama/mock）
│       └── rag_pipeline.py      # 入库与问答主流程
├── frontend/                    # Vue 3 界面
├── data/                        # 运行时数据（自动创建）
├── test_documents/              # 示例文档
├── .env                         # 本地配置（已 gitignore）
└── start.bat                    # 一键启动
```

## 快速开始

1. 安装后端依赖：

```powershell
pip install -r backend/requirements.txt
```

2. 配置模型：复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（或改用 `LLM_PROVIDER=ollama`、`mock`）。

3. 启动后端：

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

4. 启动前端：

```powershell
cd frontend
npm install
npm run dev
```

5. 打开 http://localhost:5173 ，上传文档后即可提问。API 文档见 http://localhost:8000/docs 。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查，返回当前 LLM 后端 |
| GET | `/api/documents/stats` | 知识库统计（文档数 / 分块数 / 来源） |
| POST | `/api/documents/upload` | 上传并入库（支持多文件） |
| POST | `/api/documents/rebuild` | 基于已上传文件重建索引 |
| DELETE | `/api/documents/reset` | 清空知识库与上传文件 |
| POST | `/api/chat/ask` | 同步问答，返回回答 + 引用来源 |
| POST | `/api/chat/stream` | SSE 流式问答（先推 sources，再推 token） |

## 实现要点

- **中文路径兼容**：FAISS 的 `write_index/read_index` 在 Windows 中文路径下会报 Permission denied，改用 `serialize_index` + Python 原生 I/O 读写。
- **启动速度**：所有服务与路由在 apply 阶段延迟导入，避免三方库拖慢启动。
- **离线降级**：Embedding 无法联网加载时自动使用内置字符 n-gram 向量（无需任何模型文件）；LLM 未配置时自动降级为 Mock，界面仍可完整走通流程。
- **流式容错**：部分中转站会在流末尾返回 `choices` 为空、仅含 usage 的块，已做空值保护。

## 配置项（.env）

| 变量 | 说明 |
|---|---|
| `LLM_PROVIDER` | `openai` / `deepseek` / `ollama` / `mock` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | OpenAI 兼容接口配置 |
| `DEEPSEEK_API_KEY` | DeepSeek 官方密钥（可选） |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | 本地 Ollama（可选） |
| `EMBEDDING_MODEL_NAME` | 本地 Embedding 模型名（需网络，可选） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | 分块大小、重叠、检索条数 |

## 后续可优化

- 混合检索（向量 + BM25）与 Cross-Encoder 重排序
- 多轮对话上下文记忆
- 扫描版 PDF 的 OCR 解析
- 按文档/知识库隔离多集合检索
