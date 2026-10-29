"""
FastAPI 入口 —— 延迟加载服务避免启动卡顿
启动方式: python -m uvicorn backend.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 延迟注册路由（避免 import 链卡顿）
    print("[RAG] ⏳ 注册路由...")
    from backend.routers import documents as docs_router
    from backend.routers import chat as chat_router
    app.include_router(docs_router.router, prefix="/api")
    app.include_router(chat_router.router, prefix="/api")
    print("[RAG] ✅ 路由已注册")

    # 2. 加载知识库
    print("[RAG] ⏳ 初始化知识库组件...")
    from backend.services.vector_store import get_vector_store
    store = get_vector_store()
    ok = store.load()
    stats = store.get_stats()
    print(f"[RAG] ✅ 知识库就绪 — 索引{'已加载' if ok else '新建'} {stats}")
    yield
    print("[RAG] 服务关闭")


app = FastAPI(
    title="RAG 问答系统",
    version="1.0.0",
    description="基于 LangChain + FAISS 的本地知识库问答",
    lifespan=lifespan,
)

# CORS（开发环境全开放）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
