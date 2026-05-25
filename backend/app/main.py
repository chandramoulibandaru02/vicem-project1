from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.llm_client import LLMClient
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.retriever import RetrieverService
from app.routes import chat_router, health_router, ocr_router, rag_router, search_router, upload_router, workflow_router
from app.services.document_service import document_service
from app.utils.file_utils import ensure_upload_dir


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = configure_logging()
    ensure_upload_dir(settings.upload_dir)

    app.state.settings = settings
    app.state.llm_client = LLMClient()
    app.state.retriever = RetrieverService()

    await document_service.initialize()
    await app.state.llm_client.initialize()
    await app.state.retriever.initialize()

    logger.info("ECM AI backend startup complete")
    yield

    await app.state.retriever.shutdown()
    await app.state.llm_client.shutdown()
    await document_service.shutdown()


app = FastAPI(
    title="ECM AI Platform",
    description="Integrated prototype backend for ECM AI workflows including upload, OCR, chunking, embeddings, ChromaDB search, RAG, chat, and workflow management.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(ocr_router)
app.include_router(rag_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(workflow_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "ECM AI Platform API",
        "docs": "/docs",
        "health": "/health",
    }
