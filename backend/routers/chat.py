"""
问答 API
- POST /chat/ask       : 同步问答
- POST /chat/stream    : 流式问答 (SSE)
"""
import json
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/chat", tags=["chat"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


def _get_pipeline():
    """延迟创建 RAGPipeline"""
    from backend.services.rag_pipeline import RAGPipeline
    return RAGPipeline()


@router.post("/ask")
def ask(req: AskRequest):
    """同步问答，一次性返回完整回答"""
    pipeline = _get_pipeline()
    result = pipeline.query(req.question, top_k=req.top_k)
    return result


@router.post("/stream")
def stream_ask(req: AskRequest):
    """流式问答，返回 Server-Sent Events"""
    pipeline = _get_pipeline()

    def generate():
        for chunk in pipeline.query_stream(req.question, top_k=req.top_k):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
