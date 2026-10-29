# RAG 问答系统 实现计划书

## 一、项目概述

构建一个基于 **检索增强生成 (Retrieval-Augmented Generation, RAG)** 的问答系统。
用户可以上传文档（PDF/TXT/Markdown），系统自动解析、向量化并构建知识库，
随后用户可以针对知识库内容进行自然语言提问，系统检索相关内容并由大模型生成回答。

---

## 二、技术栈选型

| 层级 | 技术 | 说明 |
|---|---|---|
| **后端框架** | Python 3.11+ + FastAPI | 高性能异步 Web 框架，自动生成 OpenAPI 文档 |
| **RAG 编排** | LangChain | 文档加载、分块、Embedding、检索、Prompt 编排 |
| **向量数据库** | FAISS (本地) | Facebook 开源，轻量级，支持 CPU 快速相似度检索，无需外部服务 |
| **Embedding 模型** | BAAI/bge-m3 (HuggingFace) | 中英双语 Embedding，多粒度检索增强，中文效果优秀 |
| **大语言模型** | 可选: DeepSeek-V3 / GPT-4o / 本地 Ollama | 通过配置切换，默认使用免费/低成本 API |
| **文档解析** | PyPDF2 / python-docx | PDF 和 Word 文档解析 |
| **前端界面** | Vue 3 + Vite + Element Plus | 现代化 SPA，支持文档上传、聊天对话、引用来源展示 |
| **进程通信** | Uvicorn (ASGI) | FastAPI 服务器 |
| **依赖管理** | pip + requirements.txt | 简单可控 |
| **数据存储** | 本地目录 (./data) | 向量索引、上传的文档、分块结果持久化 |

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户前端 (Vue 3)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ 文档上传界面 │  │ 聊天对话界面  │  │ 知识库管理面板  │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘ │
└─────────┼────────────────┼───────────────────┼─────────┘
          │ HTTP/WebSocket  │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI 后端服务                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │           RAG Pipeline (LangChain)               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │   │
│  │  │ 文档加载  │→│ 文本分块  │→│ Embedding 向量化 │ │   │
│  │  └──────────┘  └──────────┘  └───────┬────────┘ │   │
│  │                                       │          │   │
│  │                    ┌──────────────────▼────────┐ │   │
│  │                    │      FAISS 向量存储       │ │   │
│  │                    └───────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  问答流程                         │   │
│  │  用户提问 → Embedding → FAISS 相似度检索 → Prompt │   │
│  │  组装 (问题 + 上下文) → LLM 生成 → 返回回答+来源   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              外部服务 (可选/可切换)                       │
│  HuggingFace Embedding  │  LLM API (DeepSeek/GPT/本地)  │
└─────────────────────────────────────────────────────────┘
```

---

## 四、核心功能模块

### 4.1 文档处理模块
- 支持 PDF / TXT / Markdown / DOCX 格式
- 文本分块策略: 递归字符分块 (RecursiveCharacterTextSplitter)
  - chunk_size: 500 tokens, chunk_overlap: 50 tokens
- 保留元数据 (文件名、页码、块索引)

### 4.2 向量存储模块
- FAISS IndexFlatIP (内积相似度，等价于归一化后的余弦相似度)
- 向量索引持久化到 `./data/faiss_index/`
- 支持增量添加和清空重建

### 4.3 检索增强模块
- 语义相似度检索 (Top-K = 4)
- Prompt 模板: 系统指令 + 检索上下文 + 用户问题
- 回答时附带引用来源 (文件名、页码)

### 4.4 大模型接口
- 统一的 LLM 抽象层，支持:
  - DeepSeek API (推荐，性价比高)
  - OpenAI API
  - Ollama 本地模型 (完全离线)
- 流式输出支持

### 4.5 前端界面
- 文档上传区 (拖拽 + 点击)
- 知识库状态展示 (文档数量、分块数量)
- 聊天对话界面 (支持流式打字机效果)
- 回答引用来源卡片 (可点击跳转)

---

## 五、项目目录结构

```
RAG问答系统/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置 (API Key, 模型选择等)
│   ├── requirements.txt        # Python 依赖
│   ├── services/
│   │   ├── document_loader.py  # 文档加载与解析
│   │   ├── vector_store.py     # FAISS 向量存储封装
│   │   ├── embedding.py        # Embedding 模型封装
│   │   ├── llm.py              # 大模型接口封装
│   │   └── rag_pipeline.py     # RAG 主流程
│   └── routers/
│       ├── documents.py        # 文档管理 API
│       └── chat.py             # 问答 API
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── api/                # API 调用封装
│   │   ├── components/         # Vue 组件
│   │   │   ├── DocUpload.vue
│   │   │   ├── ChatPanel.vue
│   │   │   └── SourceCard.vue
│   │   └── style.css
│   └── index.html
├── data/                       # 运行时数据 (向量索引、上传文档)
├── test_documents/             # 测试文档 (README 或示例 PDF)
└── README.md
```

---

## 六、实施步骤 (按依赖顺序)

### 阶段一: 基础环境与后端核心

1. **创建项目目录结构** — backend/frontend/data 等
2. **编写 backend/requirements.txt** — 所有 Python 依赖
3. **实现 config.py** — 集中配置管理
4. **实现 embedding.py** — BGE-M3 或其他 Embedding 加载
5. **实现 document_loader.py** — 多格式文档解析
6. **实现 vector_store.py** — FAISS 封装 (init/add/search/save/load)
7. **实现 llm.py** — 统一 LLM 接口 (DeepSeek/OpenAI/Ollama)
8. **实现 rag_pipeline.py** — build_index() 和 query() 主流程
9. **实现 FastAPI routers** — documents.py + chat.py
10. **实现 main.py** — 入口、CORS、生命周期

### 阶段二: 前端界面

11. **初始化 Vue 3 + Vite 项目**
12. **配置 Element Plus**
13. **实现文档上传组件**
14. **实现聊天对话组件 (含流式响应)**
15. **实现引用来源卡片**
16. **整合所有页面 + 全局样式**

### 阶段三: 验证与优化

17. **准备测试文档**
18. **启动后端 & 前端，端到端测试**
19. **调整分块参数、检索 Top-K、Prompt 模板**

---

## 七、依赖与注意事项

| 事项 | 说明 |
|---|---|
| **Python 版本** | 建议 3.11+，FAISS 和 LangChain 对 3.8+ 兼容但推荐新版 |
| **Embedding 模型下载** | 首次运行会从 HuggingFace 下载 bge-m3 (~2GB)，可预下载到本地 |
| **Ollama 本地模型** | 如果选择完全离线方案，需先安装 Ollama 并 `ollama pull qwen2.5:7b` |
| **API Key 安全** | 通过 `.env` 文件配置，不硬编码在代码中 |
| **中文支持** | bge-m3 对中文做了专项优化，远优于 text-embedding-3-small |
| **FAISS GPU** | 默认 CPU 版本，百万级向量可考虑 faiss-gpu |

---

## 八、风险与处理

| 风险 | 影响 | 处理方案 |
|---|---|---|
| bge-m3 模型下载慢/失败 | Embedding 不可用 | 提前下载到本地或改用本地更小模型 (bge-small-zh-v1.5) |
| LLM API 限流/费用 | 问答失败/成本超支 | 默认 Ollama 本地模型，API Key 缺失时自动降级到 mock 回答以便演示 |
| FAISS 索引文件损坏 | 检索异常 | 提供重建索引 API，启动时自动检测完整性 |
| 大 PDF 解析耗时 | 用户体验差 | 异步任务 + 进度条提示 |
| 跨域问题 | 前端无法调用后端 | FastAPI 配置 CORSMiddleware，开发环境全开放 |

---

## 九、预期产出

- ✅ 可运行的后端服务 (FastAPI，端口 8000)
- ✅ 可运行的前端界面 (Vue + Vite，端口 5173)
- ✅ 支持文档上传 → 自动向量化 → 知识库问答 完整闭环
- ✅ 回答附带引用来源，可追溯
- ✅ 配置文件可切换 LLM 后端 (DeepSeek/Ollama 等)
