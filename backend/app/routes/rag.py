from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 4
    filename: str | None = None
    page_number: int | None = None


@router.post("/query", summary="Run Retrieval-Augmented Generation against indexed documents")
async def rag_query(payload: RAGQueryRequest, request: Request):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    filters = {}
    if payload.filename:
        filters["filename"] = payload.filename
    if payload.page_number is not None:
        filters["page_number"] = payload.page_number

    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG retriever is not initialized")

    return await retriever.answer_query(payload.query, top_k=payload.top_k, filters=filters or None)


@router.post("/stream", summary="Stream a Retrieval-Augmented Generation answer")
async def rag_stream(payload: RAGQueryRequest, request: Request):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    filters = {}
    if payload.filename:
        filters["filename"] = payload.filename
    if payload.page_number is not None:
        filters["page_number"] = payload.page_number

    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG retriever is not initialized")

    async def stream_generator():
        async for chunk in retriever.stream_answer(payload.query, top_k=payload.top_k, filters=filters or None):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")
