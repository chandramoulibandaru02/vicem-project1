import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.services.indexing_service import indexing_service
from app.utils.file_handler import save_upload_file

router = APIRouter(tags=["upload"])
logger = logging.getLogger("ecm_ai_backend.upload")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@router.post(
    "/upload",
    summary="Upload and index a document for ECM AI retrieval",
    description="Uploads a PDF or image, extracts text, chunks it, generates embeddings, and stores vectors in ChromaDB.",
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename = file.filename.strip()
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {suffix or 'none'}. Supported types: PDF, PNG, JPG, JPEG, BMP, TIFF."
            ),
        )

    logger.info("Received upload request", extra={"filename": filename, "content_type": file.content_type})

    try:
        settings = get_settings()
        saved_path = await save_upload_file(file, settings.upload_dir)
        result = await indexing_service.index_file(saved_path, filename)

        logger.info(
            "Upload indexing completed",
            extra={
                "filename": filename,
                "document_id": result["document_id"],
                "chunks_indexed": result["chunks_indexed"],
            },
        )

        return {
            "status": "indexed",
            "filename": filename,
            "file_path": saved_path,
            "document_id": result["document_id"],
            "page_count": result["page_count"],
            "chunks_indexed": result["chunks_indexed"],
            "embeddings_generated": result["embeddings_generated"],
            "pipeline": "upload -> OCR -> chunking -> embeddings -> ChromaDB",
        }
    except ValueError as exc:
        logger.warning("Upload indexing failed for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("Uploaded file missing during processing")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ImportError as exc:
        logger.exception("Upload dependency error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload indexing failure")
        raise HTTPException(status_code=500, detail=f"Failed to index uploaded file: {exc}") from exc
