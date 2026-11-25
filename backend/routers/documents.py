"""
文档管理 API
- POST /documents/upload  : 上传文件并入库
- POST /documents/rebuild : 重建索引
- GET  /documents/stats    : 知识库统计
- DELETE /documents/reset : 清空知识库
"""
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import UPLOAD_DIR

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIX = {".pdf", ".txt", ".md", ".docx"}


def _get_pipeline():
    """延迟创建 RAGPipeline（避免启动时 import 卡顿）"""
    from backend.services.rag_pipeline import RAGPipeline
    return RAGPipeline()


@router.get("/stats")
def get_stats():
    """获取知识库统计信息"""
    pipeline = _get_pipeline()
    store = pipeline.store

    # 按 source 统计分块数
    source_chunks: dict[str, int] = {}
    for d in store.documents:
        src = d.metadata.get("source", "未知")
        source_chunks[src] = source_chunks.get(src, 0) + 1

    uploaded_files = sorted([f.name for f in UPLOAD_DIR.iterdir() if f.is_file()])
    stats = store.get_stats()

    return {
        "message": "ok",
        "stats": stats,
        "uploaded_files": uploaded_files,
        "source_chunks": source_chunks,
    }


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传文件并自动向量化入库"""
    pipeline = _get_pipeline()
    saved_names: List[str] = []
    try:
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIX:
                raise HTTPException(400, f"不支持的文件格式: {f.filename}")

            save_path = UPLOAD_DIR / f.filename
            with open(save_path, "wb") as dst:
                shutil.copyfileobj(f.file, dst)
            saved_names.append(f.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"文件保存失败: {e}")

    # 入库
    result = pipeline.ingest_uploads(saved_names)
    return {"message": "上传成功", **result}


@router.post("/rebuild")
def rebuild_index():
    """重建整个索引"""
    pipeline = _get_pipeline()
    return pipeline.rebuild_index()


@router.delete("/reset")
def reset_knowledge():
    """清空知识库并删除所有上传文件"""
    pipeline = _get_pipeline()
    pipeline.store.reset()
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
    return {"message": "知识库已清空", "stats": pipeline.store.get_stats()}


@router.delete("/remove/{filename:path}")
def remove_document(filename: str):
    """删除单个文档（从索引和上传目录中移除）"""
    pipeline = _get_pipeline()
    removed = pipeline.store.remove_by_source(filename)

    # 删除上传的原文件
    file_path = UPLOAD_DIR / filename
    file_exists = file_path.exists()
    if file_exists:
        file_path.unlink()

    return {
        "message": "ok",
        "filename": filename,
        "chunks_removed": removed,
        "file_deleted": file_exists,
        "stats": pipeline.store.get_stats(),
    }
