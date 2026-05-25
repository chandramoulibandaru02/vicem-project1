import logging
from typing import AsyncIterator

from fastapi import HTTPException


class ChatService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend.chat")

    def _build_filters(self, filename: str | None, page_number: int | None) -> dict | None:
        filters: dict[str, str | int] = {}
        if filename:
            filters["filename"] = filename
        if page_number is not None:
            filters["page_number"] = page_number
        return filters or None

    async def answer_question(self, question: str, retriever, top_k: int = 4, filename: str | None = None, page_number: int | None = None) -> dict[str, object]:
        if retriever is None:
            raise HTTPException(status_code=503, detail="RAG retriever is not initialized")

        try:
            self.logger.info("Processing chat request", extra={"question": question[:120], "top_k": top_k})
            result = await retriever.answer_query(
                question,
                top_k=top_k,
                filters=self._build_filters(filename, page_number),
            )
            return result
        except HTTPException:
            raise
        except RuntimeError as exc:
            self.logger.exception("Runtime error while answering chat request")
            raise HTTPException(status_code=503, detail="AI chat service is temporarily unavailable") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error while answering chat request")
            raise HTTPException(status_code=500, detail="Failed to generate an answer") from exc

    async def stream_question(self, question: str, retriever, top_k: int = 4, filename: str | None = None, page_number: int | None = None) -> AsyncIterator[str]:
        if retriever is None:
            raise HTTPException(status_code=503, detail="RAG retriever is not initialized")

        self.logger.info("Starting streamed chat response", extra={"question": question[:120], "top_k": top_k})

        try:
            async for chunk in retriever.stream_answer(
                question,
                top_k=top_k,
                filters=self._build_filters(filename, page_number),
            ):
                yield chunk
        except RuntimeError as exc:
            self.logger.exception("Runtime error while streaming chat response")
            raise HTTPException(status_code=503, detail="AI chat service is temporarily unavailable") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error while streaming chat response")
            raise HTTPException(status_code=500, detail="Failed to stream an answer") from exc
