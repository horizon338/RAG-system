"""
FAISS 向量存储封装
- 使用 IndexFlatIP（内积相似度，配合归一化 embedding → 余弦相似度）
- 持久化到磁盘，启动时自动加载
"""
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from backend.config import FAISS_DIR
from .embedding import get_embeddings, embed_query
from .text_splitter import Document


_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    """
    轻量分词：
    - 英文/数字按连续串切分
    - 中文取单字 + 相邻双字（bigram），兼顾精确与召回
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
        self._loaded = False

    # ---------- 生命周期 ----------
    def load(self) -> bool:
        """从磁盘加载已有索引；返回是否加载成功"""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                # 用 Python 原生 I/O 读文件 → 绕过 FAISS C++ 层的中文路径问题
                with open(self.index_path, "rb") as f:
                    data = f.read()
                # serialize_index 产出的是 uint8 ndarray 的缓冲区，反序列化需还原成 ndarray
                self.index = faiss.deserialize_index(
                    np.frombuffer(data, dtype=np.uint8)
                )
                with open(self.meta_path, "rb") as f:
                    self.documents = pickle.load(f)
                self._loaded = True
                return True
            except Exception as e:
                print(f"[WARN] 加载 FAISS 索引失败，将重建: {e}")
        return False

    def save(self) -> None:
        """持久化索引和文档元数据"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            # 先用 FAISS 序列化到内存（bytes），再用 Python 写文件
            # 这样可以绕过 FAISS C++ 层对中文路径的兼容性问题
            data = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                f.write(data)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.documents, f)

    def reset(self) -> None:
        """清空索引"""
        self.index = None
        self.documents = []
        self._loaded = False
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def remove_by_source(self, source: str) -> int:
        """删除指定来源的所有文档分块，返回删除数量"""
        if not self.documents or self.index is None:
            return 0

        # 找出要保留的索引
        keep_indices = [
            i for i, d in enumerate(self.documents)
            if d.metadata.get("source") != source
        ]
        removed = len(self.documents) - len(keep_indices)
        if removed == 0:
            return 0

        # 重建索引 —— FAISS 没有原生删除，只能重建
        new_docs = [self.documents[i] for i in keep_indices]
        if not new_docs:
            # 删光了 → reset
            self.reset()
            return removed

        texts = [d.page_content for d in new_docs]
        vectors = np.array(get_embeddings(texts), dtype=np.float32)
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        for i, doc in enumerate(new_docs):
            doc.metadata["chunk_index"] = i
        self.documents = new_docs
        self.save()
        return removed

    # ---------- 索引操作 ----------
    def add_documents(self, docs: List[Document]) -> int:
        """向索引添加文档分块，返回本次添加的数量"""
        if not docs:
            return 0

        texts = [d.page_content for d in docs]
        vectors = np.array(get_embeddings(texts), dtype=np.float32)

        # 归一化 (保证内积 == 余弦相似度)
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        start_idx = len(self.documents)
        self.index.add(vectors)

        # 同步 chunk_index
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = start_idx + i
        self.documents.extend(docs)

        self.save()
        return len(docs)

    # ---------- 检索 ----------
    def _vector_ranking(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """向量检索，返回 [(doc_index, 余弦相似度), ...] 按分数降序"""
        q_vec = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        k = min(limit, len(self.documents))
        scores, indices = self.index.search(q_vec, k)
        out: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            out.append((int(idx), float(score)))
        return out

    def _keyword_ranking(self, query: str) -> List[Tuple[int, float]]:
        """
        BM25 关键词检索
        —— 兜底 Embedding 质量差时，靠字面匹配把相关分块抬上来
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        doc_tokens = [_tokenize(d.page_content) for d in self.documents]
        n_docs = len(doc_tokens)
        avgdl = sum(len(t) for t in doc_tokens) / n_docs or 1.0

        # 文档频率
        df: Counter = Counter()
        for tokens in doc_tokens:
            for t in set(tokens):
                df[t] += 1

        k1, b = 1.5, 0.75
        scored: List[Tuple[int, float]] = []
        for i, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            s = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1)) / (
                    tf[t] + k1 * (1 - b + b * len(tokens) / avgdl)
                )
            if s > 0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        混合检索：向量相似度 + BM25 关键词分数加权融合
        - 向量：使用真实余弦相似度（不归一化，避免噪声被放大）
        - 关键词：BM25 按本次查询最高分归一
        - 权重 向量 0.4 / 关键词 0.6（通用 Embedding 降级时关键词更可靠）
        返回 [(Document, 综合相关度 0~1), ...]
        """
        if self.index is None or not self.documents:
            return []

        vec_hits = self._vector_ranking(query, limit=max(top_k * 5, 20))
        kw_hits = self._keyword_ranking(query)

        if not vec_hits and not kw_hits:
            return []

        vec_map = {idx: max(cos, 0.0) for idx, cos in vec_hits}
        kw_map = dict(kw_hits)
        kw_max = kw_hits[0][1] if kw_hits else 0.0

        W_VEC, W_KW = 0.4, 0.6
        fused: dict[int, float] = {}
        for idx in set(vec_map) | set(kw_map):
            cos = vec_map.get(idx, 0.0)
            kw = (kw_map.get(idx, 0.0) / kw_max) if kw_max > 0 else 0.0
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
