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
    return {
        "message": "ok",
        "stats": pipeline.store.get_stats(),
        "uploaded_files": sorted([f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]),
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
