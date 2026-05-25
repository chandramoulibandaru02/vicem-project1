import logging
import re
from pathlib import Path

try:
    import fitz
    import pdfplumber
except ImportError:
    fitz = None
    pdfplumber = None


class TextExtractor:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.replace("\r\n", "\n")
        return cleaned.strip()

    def extract_pdf_text(self, file_path: str) -> dict[str, str | int]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if fitz is None or pdfplumber is None:
            raise ImportError("PyMuPDF and pdfplumber are required for PDF extraction")

        try:
            document = fitz.open(file_path)
            page_count = len(document)
            direct_chunks = []

            for page in document:
                page_text = self._clean_text(page.get_text("text"))
                if page_text:
                    direct_chunks.append(page_text)

            direct_text = "\n\n".join(direct_chunks)
            if direct_text:
                self.logger.info("PDF text extracted directly using PyMuPDF: %s", file_path)
                return {
                    "extracted_text": direct_text,
                    "page_count": page_count,
                    "extraction_method": "pdf_mupdf",
                }

            self.logger.info("No direct text detected in PDF, trying pdfplumber fallback: %s", file_path)

            with pdfplumber.open(file_path) as pdf:
                fallback_chunks = []
                for page in pdf.pages:
                    page_text = self._clean_text(page.extract_text() or "")
                    if page_text:
                        fallback_chunks.append(page_text)

            fallback_text = "\n\n".join(fallback_chunks)
            if fallback_text:
                self.logger.info("PDF text extracted using pdfplumber fallback: %s", file_path)
                return {
                    "extracted_text": fallback_text,
                    "page_count": page_count,
                    "extraction_method": "pdfplumber",
                }

            return {
                "extracted_text": "",
                "page_count": page_count,
                "extraction_method": "none",
            }
        except Exception as exc:
            self.logger.warning("Direct PDF extraction failed for %s: %s", file_path, exc)
            raise
