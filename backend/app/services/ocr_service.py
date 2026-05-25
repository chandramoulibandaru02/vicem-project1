import logging
from pathlib import Path

from app.ai.extractor import TextExtractor
from app.ai.ocr import OCRProcessor


class OCRService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.extractor = TextExtractor(self.logger)
        self.ocr_processor = OCRProcessor(self.logger)

    async def extract_document_text(self, file_path: str, filename: str) -> dict[str, str | int]:
        suffix = Path(filename).suffix.lower()

        try:
            if suffix == ".pdf":
                extraction = self.extractor.extract_pdf_text(file_path)
                page_count = int(extraction["page_count"])

                if extraction["extracted_text"]:
                    return {
                        "extracted_text": extraction["extracted_text"],
                        "page_count": page_count,
                        "ocr_status": "skipped",
                        "extraction_method": str(extraction["extraction_method"]),
                    }

                extracted_text, page_count = await self.ocr_processor.ocr_pdf(file_path)
                return {
                    "extracted_text": extracted_text,
                    "page_count": page_count,
                    "ocr_status": "performed",
                    "extraction_method": "ocr_pdf",
                }

            if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
                extracted_text = await self.ocr_processor.ocr_image(file_path)
                return {
                    "extracted_text": extracted_text,
                    "page_count": 1,
                    "ocr_status": "performed",
                    "extraction_method": "ocr_image",
                }

            raise ValueError(f"Unsupported OCR input type: {suffix}")
        except Exception as exc:
            self.logger.warning("OCR pipeline failed for %s: %s", filename, exc)
            return {
                "extracted_text": "",
                "page_count": 1,
                "ocr_status": "failed",
                "extraction_method": "ocr_failed",
            }


ocr_service = OCRService()
