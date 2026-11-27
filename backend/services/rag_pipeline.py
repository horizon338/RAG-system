"""
RAG 主流程编排 —— 基于 LangChain LCEL
- 入库：加载 → 切分 → Embedding → FAISS
- 问答：混合检索 → ChatPromptTemplate → ChatModel → StrOutputParser
- 支持多轮对话历史（MessagesPlaceholder）
"""
from pathlib import Path
from typing import Any, Dict, Generator, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.config import settings, UPLOAD_DIR
from .document_loader import load_and_split_documents
from .llm import get_chat_model
from .vector_store import VectorStore, get_vector_store

# 历史消息最多保留条数（user + assistant 各算一条）
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

HUMAN_TEMPLATE = """{kb_list}【参考资料】
{context}

---

【用户问题】
{question}"""


def _truncate_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """保留最近 N 条历史，避免 token 爆炸"""
    if not history:
        return []
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    if trimmed and trimmed[0].get("role") == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def _to_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """把前端传来的 history 转成 LangChain 消息对象"""
    msgs: List[BaseMessage] = []
    for item in _truncate_history(history):
        if item.get("role") == "assistant":
            msgs.append(AIMessage(content=item.get("content", "")))
        else:
            msgs.append(HumanMessage(content=item.get("content", "")))
    return msgs


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, llm=None):
        self.store = store or get_vector_store()
        self.llm = llm or get_chat_model()

        # LCEL 链：Prompt → ChatModel → 纯文本输出
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history", optional=True),
                ("human", HUMAN_TEMPLATE),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ---------- 索引构建 ----------
    def ingest_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """上传并向量化文档"""
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
    def _format_context(self, context_docs: list) -> str:
        parts: List[str] = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            header = f"[资料{i}] {src}" + (f" (第{page}页)" if page else "")
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def _kb_list(self) -> str:
        """知识库文档清单（与检索结果解耦，覆盖全部文档）"""
        stats = self.store.get_stats()
        if not stats.get("sources"):
            return ""
        lines = [
            f"- {name}（{stats['source_chunks'].get(name, 0)} 个片段）"
            for name in stats["sources"]
        ]
        return "【知识库文档清单】\n" + "\n".join(lines) + "\n\n"

    def _build_payload(
        self,
        question: str,
        context_docs: list,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        return {
            "kb_list": self._kb_list(),
            "context": self._format_context(context_docs),
            "question": question,
            "history": _to_messages(history or []),
        }

    def _collect_sources(self, hits: list) -> List[Dict[str, Any]]:
        return [
            {
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
                "score": round(score, 4),
                "preview": doc.page_content[:150],
            }
            for doc, score in hits
        ]

    def query(
        self,
        question: str,
        top_k: int | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """同步 RAG 问答，返回回答 + 引用来源"""
        top_k = top_k or settings.top_k

        if not self.store.is_ready:
            return {"answer": "知识库为空，请先上传文档！", "sources": [], "retrieved_chunks": 0}

        hits = self.store.search(question, top_k=top_k)
        if not hits:
            return {
                "answer": "检索失败，请检查 Embedding 或索引是否正常。",
                "sources": [],
                "retrieved_chunks": 0,
            }

        payload = self._build_payload(question, [d for d, _ in hits], history)
        answer = self.chain.invoke(payload)

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
        流式 RAG 问答：先推 sources，再由 LCEL 链逐步推 token，最后 done
        """
        top_k = top_k or settings.top_k

        if not self.store.is_ready:
            yield {"type": "error", "content": "知识库为空，请先上传文档！"}
            return

        hits = self.store.search(question, top_k=top_k)
        yield {"type": "sources", "sources": self._collect_sources(hits)}

        payload = self._build_payload(question, [d for d, _ in hits], history)
        try:
            for delta in self.chain.stream(payload):
                if delta:
                    yield {"type": "token", "content": delta}
        except Exception as e:
            yield {"type": "error", "content": f"LLM 调用失败: {e}"}

        yield {"type": "done"}
