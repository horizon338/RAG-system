"""
RAG 主流程编排
- build_index(): 上传文档 → 解析 → 分块 → Embedding → FAISS
- query(): 用户问题 → 检索 → Prompt 组装 → LLM 生成回答
- 支持多轮对话历史传递
"""
from pathlib import Path
from typing import List, Dict, Any, Generator

from backend.config import settings, UPLOAD_DIR
from .document_loader import load_and_split_documents
from .vector_store import VectorStore, get_vector_store
from .llm import get_llm_client, BaseLLM

# 最多保留的历史消息条数（user + assistant 各算一条）
MAX_HISTORY_MESSAGES = 6


SYSTEM_PROMPT = """你是一个专业的问答助手。请基于提供的【参考资料】来回答用户的问题。

规则:
1. 优先使用参考资料中的信息，不要编造答案
2. 如果参考资料中没有足够信息，请明确说明"根据现有知识库无法回答这个问题"
3. 回答要结构清晰、语言流畅，适合中文阅读
4. 回答中如有引用参考资料，请在文末标注来源文件名
5. 可以结合对话上下文理解代词（如"它"、"这个"、"刚才说的"等），但回答仍以当前参考资料为准
6. 当用户询问"知识库里有哪些文档/有没有某类文件"时，请依据【知识库文档清单】如实回答，不要声称没有
   （注意：检索只取出最相关的若干片段，没被检索到不代表知识库中没有该文档）
"""


def _truncate_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """保留最近的 N 条消息，避免 token 爆炸"""
    if not history:
        return []
    # 确保最后一条是 user（如果有 assistant 结尾则保留前面的）
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    # 如果被截断后第一条是 assistant，丢弃它让 user 开头
    if trimmed and trimmed[0].get("role") == "assistant":
        trimmed = trimmed[1:]
    return trimmed


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, llm: BaseLLM | None = None):
        self.store = store or get_vector_store()
        self.llm = llm or get_llm_client()

    # ---------- 索引构建 ----------
    def ingest_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        上传并向量化文档
        返回: 统计信息
        """
        splits = load_and_split_documents(file_paths)
        added = self.store.add_documents(splits)
        return {
            "files": [p.name for p in file_paths],
            "chunks_added": added,
            "total_chunks": self.store.chunk_count,
            "stats": self.store.get_stats(),
        }

    def ingest_uploads(self, filenames: List[str]) -> Dict[str, Any]:
        """根据已保存到 UPLOAD_DIR 的文件名入库"""
        paths: List[Path] = []
        for name in filenames:
            p = UPLOAD_DIR / name
            if p.exists():
                paths.append(p)
        return self.ingest_files(paths)

    def rebuild_index(self) -> Dict[str, Any]:
        """扫描 UPLOAD_DIR 重建整个索引"""
        self.store.reset()
        files = [p for p in UPLOAD_DIR.iterdir() if p.is_file()]
        if not files:
            return {"message": "没有上传的文件，索引已清空", "chunks": 0}
        result = self.ingest_files(files)
        result["message"] = "索引已重建"
        return result

    # ---------- 问答 ----------
    def _build_messages(
        self,
        question: str,
        context_docs: list,
        history: List[Dict[str, str]] | None = None,
    ) -> List[Dict[str, str]]:
        context_parts: List[str] = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            header = f"[资料{i}] {src}" + (f" (第{page}页)" if page else "")
            context_parts.append(f"{header}\n{doc.page_content}")

        context_str = "\n\n".join(context_parts)

        # 知识库文档清单 —— 让模型知道库里到底有哪些文件
        kb_list = ""
        stats = self.store.get_stats()
        if stats.get("sources"):
            lines = [
                f"- {name}（{stats['source_chunks'].get(name, 0)} 个片段）"
                for name in stats["sources"]
            ]
            kb_list = "【知识库文档清单】\n" + "\n".join(lines) + "\n\n"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # 注入历史对话（截断后）
        if history:
            messages.extend(_truncate_history(history))

        # 当前用户问题 + 参考资料
        user_content = (
            f"{kb_list}"
            f"【参考资料】\n{context_str}\n\n"
            f"---\n\n"
            f"【用户问题】\n{question}"
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    def _collect_sources(self, hits: list) -> List[Dict[str, Any]]:
        sources = []
        for doc, score in hits:
            sources.append({
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
                "score": round(score, 4),
                "preview": doc.page_content[:150],
            })
        return sources

    def query(
        self,
        question: str,
        top_k: int | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """
        完整 RAG 查询
        返回: {"answer": str, "sources": [...], "retrieved_chunks": int}
        """
        top_k = top_k or settings.top_k

        if not self.store.is_ready:
            return {
                "answer": "知识库为空，请先上传文档！",
                "sources": [],
                "retrieved_chunks": 0,
            }

        hits = self.store.search(question, top_k=top_k)
        if not hits:
            return {
                "answer": "检索失败，请检查 Embedding 模型是否正常加载。",
                "sources": [],
                "retrieved_chunks": 0,
            }

        context_docs = [doc for doc, _ in hits]
        messages = self._build_messages(question, context_docs, history)
        answer = self.llm.chat(messages)

        return {
            "answer": answer,
            "sources": self._collect_sources(hits),
            "retrieved_chunks": len(hits),
        }

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式 RAG 查询（Yield 字典）
        - 先 yield 检索到的 sources
        - 再逐步 yield 回答片段
        - 最后 yield done=True
        """
        top_k = top_k or settings.top_k

        if not self.store.is_ready:
            yield {"type": "error", "content": "知识库为空，请先上传文档！"}
            return

        hits = self.store.search(question, top_k=top_k)
        context_docs = [doc for doc, _ in hits]

        yield {"type": "sources", "sources": self._collect_sources(hits)}

        messages = self._build_messages(question, context_docs, history)
        try:
            for delta in self.llm.chat_stream(messages):
                yield {"type": "token", "content": delta}
        except Exception as e:
            yield {"type": "error", "content": f"LLM 调用失败: {e}"}

        yield {"type": "done"}
