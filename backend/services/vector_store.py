"""
FAISS 向量存储封装
- 使用 IndexFlatIP（内积相似度，配合归一化 embedding → 余弦相似度）
- 持久化到磁盘，启动时自动加载
"""
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from backend.config import FAISS_DIR
from .embedding import get_embeddings, embed_query
from .text_splitter import Document


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
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Document, float]]:
        """
        语义检索，返回 [(Document, 相似度), ...]
        """
        if self.index is None or not self.documents:
            return []

        q_vec = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q_vec, k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append((self.documents[idx], float(score)))
        return results

    # ---------- 状态查询 ----------
    @property
    def chunk_count(self) -> int:
        return len(self.documents)

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.documents) > 0

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        sources = set()
        for d in self.documents:
            if "source" in d.metadata:
                sources.add(d.metadata["source"])
        return {
            "documents": len(sources),
            "chunks": len(self.documents),
            "ready": self.is_ready,
            "sources": sorted(sources),
        }


# 全局单例
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store
