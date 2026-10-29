"""
极简文档分块器（不依赖 LangChain）
实现 RecursiveCharacterTextSplitter 的核心逻辑
"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Document:
    """最小化 Document，兼容 page_content + metadata"""
    page_content: str = ""
    metadata: Dict = field(default_factory=dict)


DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def _split_text(text: str, sep: str) -> List[str]:
    if not sep:
        # 字符级分割
        return [c for c in text]
    return text.split(sep)


def _merge_splits(splits: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """将切好的碎片合并为接近 chunk_size 的块"""
    chunks: List[str] = []
    current = []
    cur_len = 0

    for s in splits:
        s_len = len(s)
        if cur_len + s_len > chunk_size and current:
            chunks.append("".join(current))
            # overlap: 保留最后一段
            overlap_chars = chunk_overlap
            overlap_text = "".join(current)[-overlap_chars:] if overlap_chars > 0 else ""
            current = [overlap_text] if overlap_text else []
            cur_len = len(overlap_text)
        current.append(s)
        cur_len += s_len

    if current:
        chunks.append("".join(current))

    return [c for c in chunks if c.strip()]


def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[str]:
    """递归分块：用分隔符列表依次尝试，直到块足够小"""
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # 如果文本已经够小，直接返回
    if len(text) <= chunk_size:
        return [text]

    # 依次尝试分隔符
    for sep in separators:
        parts = _split_text(text, sep)
        # 如果分割后至少有一块（说明这个分隔符有效）
        if len(parts) > 1 or (sep == "" and len(text) > 0):
            # 递归处理每一块
            final: List[str] = []
            for p in parts:
                if len(p) > chunk_size:
                    final.extend(recursive_split(p, chunk_size, chunk_overlap, separators))
                else:
                    final.append(p)
            # 合并成接近 chunk_size 的块
            return _merge_splits(final, chunk_size, chunk_overlap)

    # 兜底：直接按字符切
    return _merge_splits(list(text), chunk_size, chunk_overlap)


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """将 Document 列表分块"""
    result: List[Document] = []
    for doc in docs:
        chunks = recursive_split(doc.page_content, chunk_size, chunk_overlap)
        for chunk in chunks:
            meta = dict(doc.metadata)  # 拷贝
            result.append(Document(page_content=chunk, metadata=meta))
    return result
