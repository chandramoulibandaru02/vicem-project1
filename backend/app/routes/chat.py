from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="User question to send to the ECM AI assistant")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of semantic retrieval chunks to use")
    filename: str | None = Field(default=None, description="Optional filename filter for semantic retrieval")
    page_number: int | None = Field(default=None, ge=1, description="Optional page number filter for semantic retrieval")


class SourceItem(BaseModel):
    filename: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    parent_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask the ECM AI assistant a question",
    description="Uses semantic retrieval plus Groq to answer user questions with citations.",
)
async def post_chat(payload: ChatRequest, request: Request):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    retriever = getattr(request.app.state, "retriever", None)
    result = await chat_service.answer_question(
        question=payload.question,
        retriever=retriever,
        top_k=payload.top_k,
        filename=payload.filename,
        page_number=payload.page_number,
    )

    return ChatResponse(
        answer=str(result.get("answer", "")),
        sources=[SourceItem(**source) for source in result.get("sources", [])],
    )


@router.post(
    "/stream",
    summary="Stream the ECM AI assistant response",
    description="Streams the answer and citations for a question using the same semantic retrieval pipeline.",
)
async def post_chat_stream(payload: ChatRequest, request: Request):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    retriever = getattr(request.app.state, "retriever", None)

    async def stream_generator():
        async for chunk in chat_service.stream_question(
            question=payload.question,
            retriever=retriever,
            top_k=payload.top_k,
            filename=payload.filename,
            page_number=payload.page_number,
        ):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")
