"""
文本分块 —— 基于 LangChain
- Document 复用 langchain_core.documents.Document
- 分块使用 RecursiveCharacterTextSplitter（中文标点优先切分）
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文场景的分隔符优先级：段落 → 换行 → 句末标点 → 分句标点 → 空格 → 字符
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def get_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,
    )


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """将 Document 列表切分为分块列表（保留原有 metadata）"""
    splitter = get_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
