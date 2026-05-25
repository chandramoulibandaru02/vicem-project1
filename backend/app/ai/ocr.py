import asyncio
import logging
import re
import tempfile
from pathlib import Path

try:
    import fitz
    from paddleocr import PaddleOCR
except ImportError:
    fitz = None
    PaddleOCR = None


class OCRProcessor:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self._ocr_engine = None

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.replace("\r\n", "\n")
        return cleaned.strip()

    def _get_ocr_engine(self) -> PaddleOCR:
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is required for OCR processing")

        if self._ocr_engine is None:
            self.logger.info("Initializing PaddleOCR engine")
            self._ocr_engine = PaddleOCR(
                lang="en",
                use_angle_cls=False,
                use_doc_orientation_classify=False,
                use_textline_orientation=False,
            )
        return self._ocr_engine

    def _parse_ocr_result(self, result) -> str:
        if not result or not result[0]:
            return ""

        lines = []
        for item in result[0]:
            text = item[1][0]
            if text:
                lines.append(text)

        return self._clean_text("\n".join(lines))

    def _ocr_single_image(self, image_path: str) -> str:
        try:
            result = self._get_ocr_engine().ocr(image_path, cls=False)
            return self._parse_ocr_result(result)
        except Exception as exc:
            self.logger.warning("OCR failed for image %s: %s", image_path, exc)
            raise

    async def ocr_image(self, image_path: str) -> str:
        return await asyncio.to_thread(self._ocr_single_image, image_path)

    async def ocr_pdf(self, file_path: str) -> tuple[str, int]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF OCR")

        try:
            document = fitz.open(file_path)
            page_count = len(document)
            page_texts = []

            for page_index in range(page_count):
                page = document.load_page(page_index)
                pix = page.get_pixmap(dpi=150)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name

                try:
                    pix.save(temp_path)
                    page_text = await self.ocr_image(temp_path)
                    if page_text:
                        page_texts.append(page_text)
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            return self._clean_text("\n\n".join(page_texts)), page_count
        except Exception as exc:
            self.logger.warning("OCR failed for PDF %s: %s", file_path, exc)
            raise
