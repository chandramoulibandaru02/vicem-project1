from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.services.ocr_service import ocr_service
from app.utils.file_handler import save_upload_file

router = APIRouter(prefix="/ocr", tags=["ocr"])
SUPPORTED_PDF = ".pdf"
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@router.post("/extract", summary="Extract text from PDF or image using direct extraction and OCR")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix.lower()

    if suffix != SUPPORTED_PDF and suffix not in SUPPORTED_IMAGES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Supported inputs: PDF, PNG, JPG, JPEG, BMP, TIFF.",
        )

    try:
        settings = get_settings()
        saved_path = await save_upload_file(file, settings.upload_dir)
        result = await ocr_service.extract_document_text(saved_path, file.filename)

        return {
            "filename": file.filename,
            "file_path": saved_path,
            "extracted_text": result["extracted_text"],
            "page_count": result["page_count"],
            "ocr_status": result["ocr_status"],
            "extraction_method": result["extraction_method"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="OCR processing failed")
