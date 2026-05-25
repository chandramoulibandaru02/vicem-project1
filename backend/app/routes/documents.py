from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_service import document_service
from app.utils.file_utils import save_upload_file

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", summary="Upload a document for ECM processing")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    saved_path = await save_upload_file(file)
    status = await document_service.register_upload(saved_path, file.filename)

    return {
        "filename": file.filename,
        "path": saved_path,
        "status": status,
    }
