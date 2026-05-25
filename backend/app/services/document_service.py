import logging
from pathlib import Path


class DocumentService:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ecm_ai_backend")

    async def initialize(self) -> None:
        self.logger.info("Document service initialized")

    async def shutdown(self) -> None:
        self.logger.info("Document service shutdown")

    async def process_upload(self, saved_path: str, filename: str) -> dict[str, str]:
        path = Path(saved_path)

        if not path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {saved_path}")

        size = path.stat().st_size
        self.logger.info("Processed upload: %s -> %s (%s bytes)", filename, saved_path, size)

        return {"processing_status": "pending_processing"}


document_service = DocumentService()
