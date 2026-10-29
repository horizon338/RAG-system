"""
Embedding 模型封装
优先使用 BAAI/bge-small-zh-v1.5 (通过 HF 镜像)
网络不可用时 fallback 到 TF-IDF 风格的简易向量，保证系统可演示
"""
import hashlib
import math
from collections import Counter
from typing import List
import numpy as np

from backend.config import settings

_embedding_model = None
_embedding_fallback = False  # True = 使用 fallback 简易向量


def _hash_char(c: str) -> int:
    """字符 hash 到一个维度"""
    return int(hashlib.md5(c.encode("utf-8")).hexdigest(), 16)


class _FallbackEmbedder:
    """简易 TF-IDF 风格向量 —— 不依赖任何外部模型"""

    DIM = 384  # 维度跟 bge-small-zh 对齐

    @staticmethod
    def _vectorize(texts: List[str]) -> np.ndarray:
        """将文本转为简单的字符 n-gram 频率向量"""
        vectors = np.zeros((len(texts), _FallbackEmbedder.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # 取 2-gram 和 3-gram
            grams = []
            for j in range(len(text) - 1):
                grams.append(text[j : j + 2])
            for j in range(len(text) - 2):
                grams.append(text[j : j + 3])
            # ASCII 空格分词
            for w in text.split():
                grams.append(w)

            total = len(grams) or 1
            for g, cnt in Counter(grams).items():
                # hash 到维度
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % _FallbackEmbedder.DIM
                vectors[i, idx] += cnt / total

        # L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors = vectors / norms
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._vectorize(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize([text])[0].tolist()


# 设置为 True 可尝试加载 HuggingFace 模型（需要网络）
TRY_HF = False


def _try_load_hf():
    """尝试加载 HuggingFace Embedding，失败时返回 None"""
    if not TRY_HF:
        print("[Embedding] ⏭️ 跳过 HF 模型加载，使用 fallback 简易向量")
        return None

    try:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from langchain_huggingface import HuggingFaceEmbeddings

        print(f"[Embedding] 📥 正在加载 {settings.embedding_model_name} ...")
        model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        _ = model.embed_query("test")
        print(f"[Embedding] ✅ 已加载 {settings.embedding_model_name}")
        return model
    except Exception as e:
        print(f"[Embedding] ⚠️ HF 模型加载失败: {e}")
        print("[Embedding] 🔁 使用 fallback 简易向量（演示用，效果有限）")
        return None


def _get_model():
    global _embedding_model, _embedding_fallback
    if _embedding_model is None:
        hf = _try_load_hf()
        if hf is not None:
            _embedding_model = hf
            _embedding_fallback = False
        else:
            _embedding_model = _FallbackEmbedder()
            _embedding_fallback = True
    return _embedding_model


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """批量将文本列表转为向量"""
    model = _get_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """将单条查询文本转为向量"""
    model = _get_model()
    return model.embed_query(text)


def is_fallback() -> bool:
    """返回当前是否使用 fallback 简易向量"""
    _get_model()
    return _embedding_fallback


def reload_embedding_model() -> None:
    """强制重新加载"""
    global _embedding_model, _embedding_fallback
    _embedding_model = None
    _embedding_fallback = False
    _get_model()
