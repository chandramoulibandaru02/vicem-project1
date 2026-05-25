import logging

from app.core.config import get_settings
from app.rag.pipeline import RAGPipeline


class RetrieverService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.pipeline: RAGPipeline | None = None

    async def initialize(self) -> None:
        settings = get_settings()
        self.pipeline = RAGPipeline(
            logger=self.logger,
            groq_api_key=settings.groq_api_key or settings.model_api_key,
            groq_model=settings.groq_model,
        )
        self.logger.info("RAG retriever service initialized")

    async def shutdown(self) -> None:
        self.logger.info("RAG retriever service shutdown")
        self.pipeline = None

    async def retrieve_context(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self.pipeline is None:
            raise RuntimeError("RetrieverService is not initialized")

        return await self.pipeline.retrieve_context(query, top_k=top_k, filters=filters)

    async def search(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self.pipeline is None:
            raise RuntimeError("RetrieverService is not initialized")

        return await self.pipeline.search_documents(query, top_k=top_k, filters=filters)

    async def answer_query(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self.pipeline is None:
            raise RuntimeError("RetrieverService is not initialized")

        return await self.pipeline.answer_query(query, top_k=top_k, filters=filters)

    async def stream_answer(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self.pipeline is None:
            raise RuntimeError("RetrieverService is not initialized")

        return self.pipeline.stream_answer(query, top_k=top_k, filters=filters)
