from .chat import router as chat_router
from .health import router as health_router
from .ocr import router as ocr_router
from .rag import router as rag_router
from .search import router as search_router
from .upload import router as upload_router
from .workflow import router as workflow_router

__all__ = ["chat_router", "health_router", "ocr_router", "rag_router", "search_router", "upload_router", "workflow_router"]
