import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

import fitz

from app.ai.ocr import OCRProcessor
from app.rag.chunker import create_chunk_records
from app.rag.embeddings import EmbeddingGenerator
from app.rag.vector_store import VectorStore


class IndexingService:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        ocr_processor: OCRProcessor | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend.indexing")
        self.vector_store = vector_store or VectorStore(logger=self.logger)
        self.embedding_generator = embedding_generator or EmbeddingGenerator(logger=self.logger)
        self.ocr_processor = ocr_processor or OCRProcessor(self.logger)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "\n\n".join(part.strip() for part in cleaned.split("\n\n") if part.strip())
        return cleaned.strip()

    async def _extract_pdf_pages(self, file_path: str, filename: str) -> list[tuple[int, str]]:
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF indexing")

        pages: list[tuple[int, str]] = []
        document = fitz.open(file_path)

        try:
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                page_text = self._clean_text(page.get_text("text"))

                if not page_text:
                    pix = page.get_pixmap(dpi=140)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_path = temp_file.name
                    try:
                        pix.save(temp_path)
                        page_text = await self.ocr_processor.ocr_image(temp_path)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

                page_text = self._clean_text(page_text)
                if page_text:
                    pages.append((page_index + 1, page_text))
                else:
                    self.logger.warning("No text could be extracted from page %s of %s", page_index + 1, filename)

            return pages
        finally:
            document.close()

    async def _extract_pdf_text(self, file_path: str, filename: str) -> list[tuple[int, str]]:
        pages = await self._extract_pdf_pages(file_path, filename)

        if not pages:
            raise ValueError(f"No extractable text found in PDF: {filename}")

        return pages

    async def _extract_image_text(self, file_path: str, filename: str) -> list[tuple[int, str]]:
        text = await self.ocr_processor.ocr_image(file_path)
        text = self._clean_text(text)
        if not text:
            raise ValueError(f"No extractable text found in image: {filename}")
        return [(1, text)]

    async def _build_index_records(self, file_path: str, filename: str) -> tuple[list[dict[str, str | int | None]], int]:
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            page_texts = await self._extract_pdf_text(file_path, filename)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            page_texts = await self._extract_image_text(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type for indexing: {suffix}")

        chunks: list[dict[str, str | int | None]] = []
        for page_number, page_text in page_texts:
            page_chunks = create_chunk_records(page_text, filename, page_number, logger=self.logger)["parents"]
            chunks.extend(page_chunks)

        self.logger.info("Prepared %s chunks for %s", len(chunks), filename)
        return chunks, len(page_texts)

    async def index_file(self, file_path: str, filename: str) -> dict[str, object]:
        document_id = uuid.uuid4().hex
        self.logger.info("Starting document indexing", extra={"document_id": document_id, "filename": filename})

        chunks, page_count = await self._build_index_records(file_path, filename)

        if not chunks:
            raise ValueError(f"No chunks were generated for {filename}")

        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = await asyncio.to_thread(self.embedding_generator.generate_embeddings, texts)
        embedding_list = embeddings.astype(float).tolist()

        metadatas = []
        ids = []
        for chunk in chunks:
            metadata = dict(chunk["metadata"])
            metadata["document_id"] = document_id
            metadata["source_path"] = file_path
            metadata["filename"] = filename
            metadatas.append(metadata)
            ids.append(str(metadata["chunk_id"]))

        self.vector_store.add_documents(
            collection_name="ecm_documents",
            documents=texts,
            embeddings=embedding_list,
            metadatas=metadatas,
            ids=ids,
        )

        response = {
            "document_id": document_id,
            "filename": filename,
            "status": "indexed",
            "page_count": page_count,
            "chunks_indexed": len(chunks),
            "embeddings_generated": len(embedding_list),
            "source_path": file_path,
        }

        self.logger.info("Completed document indexing", extra=response)
        return response


indexing_service = IndexingService()
