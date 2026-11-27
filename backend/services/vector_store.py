"""
向量存储 —— 基于 LangChain
- 向量库：langchain_community.vectorstores.FAISS（IndexFlatIP + 余弦相似度）
- 关键词：langchain_community.retrievers.BM25Retriever（rank_bm25 后端）
- 混合检索：两路分数加权融合
- 持久化：FAISS serialize_index + Python 原生 I/O，绕过 Windows 中文路径问题
"""
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from backend.config import FAISS_DIR
from .embedding import get_embeddings, embed_query, get_lc_embeddings


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 向量 / 关键词 权重（默认 Embedding 为降级向量，关键词更可信）
W_VEC = 0.4
W_KW = 0.6


def tokenize(text: str) -> List[str]:
    """
    BM25 分词函数：
    - 英文 / 数字按连续串切分
    - 中文取单字 + 相邻双字（bigram）
    """
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    cjk = _CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


class VectorStore:
    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or FAISS_DIR / "index.faiss"
        self.meta_path = (index_path or FAISS_DIR / "index.faiss").with_suffix(".meta.pkl")
        self.index: faiss.IndexFlatIP | None = None
        self.documents: List[Document] = []
        # LangChain 组件（随 documents 变化重建）
        self.vectorstore: FAISS | None = None
        self.bm25: BM25Retriever | None = None
        self._loaded = False

    # ---------- LangChain 组件装配 ----------
    def _rebuild_retrievers(self) -> None:
        """根据当前 documents / index 重建 FAISS 与 BM25Retriever"""
        if self.index is None or not self.documents:
            self.vectorstore = None
            self.bm25 = None
            return

        docstore = InMemoryDocstore(
            {str(i): d for i, d in enumerate(self.documents)}
        )
        index_to_id = {i: str(i) for i in range(len(self.documents))}
        self.vectorstore = FAISS(
            embedding_function=get_lc_embeddings(),
            index=self.index,
            docstore=docstore,
            index_to_docstore_id=index_to_id,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        self.bm25 = BM25Retriever.from_documents(
            self.documents, preprocess_func=tokenize
        )
        self.bm25.k = min(len(self.documents), 20)

    # ---------- 生命周期 ----------
    def load(self) -> bool:
        """从磁盘加载已有索引；返回是否加载成功"""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                # Python 原生 I/O 绕过 FAISS C++ 层的中文路径问题
                with open(self.index_path, "rb") as f:
                    data = f.read()
                self.index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
                with open(self.meta_path, "rb") as f:
                    self.documents = pickle.load(f)
                self._rebuild_retrievers()
                self._loaded = True
                return True
            except Exception as e:
                print(f"[WARN] 加载索引失败，将重建: {e}")
                self.index = None
                self.documents = []
        return False

    def save(self) -> None:
        """持久化索引与分块元数据"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            data = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                f.write(data)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.documents, f)

    def reset(self) -> None:
        """清空索引"""
        self.index = None
        self.documents = []
        self.vectorstore = None
        self.bm25 = None
        self._loaded = False
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def remove_by_source(self, source: str) -> int:
        """删除指定来源的所有分块，返回删除数量"""
        if not self.documents or self.index is None:
            return 0

        keep_indices = [
            i for i, d in enumerate(self.documents)
            if d.metadata.get("source") != source
        ]
        removed = len(self.documents) - len(keep_indices)
        if removed == 0:
            return 0

        new_docs = [self.documents[i] for i in keep_indices]
        if not new_docs:
            self.reset()
            return removed

        self._index_from_scratch(new_docs)
        return removed

    def _index_from_scratch(self, docs: List[Document]) -> None:
        """对给定分块重新向量化并重建索引"""
        texts = [d.page_content for d in docs]
        vectors = np.array(get_embeddings(texts), dtype=np.float32)
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
        self.documents = docs
        self._rebuild_retrievers()
        self.save()

    # ---------- 索引操作 ----------
    def add_documents(self, docs: List[Document]) -> int:
        """向索引追加分块，返回本次添加数量"""
        if not docs:
            return 0

        texts = [d.page_content for d in docs]
        vectors = np.array(get_embeddings(texts), dtype=np.float32)
        faiss.normalize_L2(vectors)

        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])

        start_idx = len(self.documents)
        self.index.add(vectors)

        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = start_idx + i
        self.documents.extend(docs)

        self._rebuild_retrievers()
        self.save()
        return len(docs)

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        混合检索（LangChain FAISS + BM25Retriever）
        分数 = 0.4 × 余弦相似度 + 0.6 × (BM25 / 本次最高 BM25)
        返回 [(Document, 综合相关度 0~1), ...]
        """
        if not self.is_ready or self.vectorstore is None:
            return []

        # 1) 向量路：IndexFlatIP 的 score 即内积（越高越相似）
        n_candidates = min(max(top_k * 5, 20), len(self.documents))
        vec_pairs = self.vectorstore.similarity_search_with_score(query, k=n_candidates)

        # 用对象身份定位回 self.documents 下标
        id_to_idx = {id(d): i for i, d in enumerate(self.documents)}
        vec_scores: dict[int, float] = {}
        for doc, score in vec_pairs:
            idx = id_to_idx.get(id(doc))
            if idx is not None:
                vec_scores[idx] = max(float(score), 0.0)

        # 2) 关键词路：BM25Okapi 打分
        kw_scores: dict[int, float] = {}
        if self.bm25 is not None and getattr(self.bm25, "vectorizer", None) is not None:
            raw = self.bm25.vectorizer.get_scores(tokenize(query))
            for i, s in enumerate(raw):
                if s > 0:
                    kw_scores[i] = float(s)

        if not vec_scores and not kw_scores:
            return []

        # 3) 融合
        kw_max = max(kw_scores.values()) if kw_scores else 0.0
        fused: dict[int, float] = {}
        for idx in set(vec_scores) | set(kw_scores):
            cos = vec_scores.get(idx, 0.0)
            kw = (kw_scores.get(idx, 0.0) / kw_max) if kw_max > 0 else 0.0
            fused[idx] = W_VEC * cos + W_KW * kw

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[idx], round(score, 4)) for idx, score in ranked]

    # ---------- 状态查询 ----------
    @property
    def chunk_count(self) -> int:
        return len(self.documents)

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.documents) > 0

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        source_chunks: Counter = Counter()
        for d in self.documents:
            src = d.metadata.get("source")
            if src:
                source_chunks[src] += 1
        return {
            "documents": len(source_chunks),
            "chunks": len(self.documents),
            "ready": self.is_ready,
            "sources": sorted(source_chunks.keys()),
            "source_chunks": dict(source_chunks),
        }


# 全局单例
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store
