import logging
import uuid
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def build_chunk_metadata(
    filename: str,
    page_number: int,
    chunk_id: str,
    parent_id: str | None = None,
) -> dict[str, str | int | None]:
    return {
        "filename": filename,
        "page_number": page_number,
        "chunk_id": chunk_id,
        "parent_id": parent_id,
    }


def create_parent_chunks(
    text: str,
    filename: str,
    page_number: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    logger: logging.Logger | None = None,
) -> list[dict[str, str | int | None]]:
    logger = logger or logging.getLogger("ecm_ai_backend")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    parent_texts = splitter.split_text(text)
    chunks = []

    for index, chunk_text in enumerate(parent_texts, start=1):
        chunk_id = f"parent-{page_number}-{index}-{uuid.uuid4().hex[:8]}"
        chunks.append(
            {
                "text": chunk_text,
                "metadata": build_chunk_metadata(filename, page_number, chunk_id),
                "level": "parent",
            }
        )

    logger.info(
        "Created %s parent chunks for %s page %s",
        len(chunks),
        filename,
        page_number,
    )
    return chunks


def create_child_chunks(
    parent_chunks: Iterable[dict[str, str | int | None]],
    filename: str,
    page_number: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    logger: logging.Logger | None = None,
) -> list[dict[str, str | int | None]]:
    logger = logger or logging.getLogger("ecm_ai_backend")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    child_chunks = []

    for parent_chunk in parent_chunks:
        parent_id = str(parent_chunk["metadata"]["chunk_id"])
        parent_text = str(parent_chunk["text"])
        child_texts = splitter.split_text(parent_text)

        for index, child_text in enumerate(child_texts, start=1):
            child_id = f"child-{page_number}-{index}-{uuid.uuid4().hex[:8]}"
            child_chunks.append(
                {
                    "text": child_text,
                    "metadata": build_chunk_metadata(
                        filename,
                        page_number,
                        child_id,
                        parent_id=parent_id,
                    ),
                    "level": "child",
                    "parent_id": parent_id,
                }
            )

    logger.info(
        "Created %s child chunks for %s page %s",
        len(child_chunks),
        filename,
        page_number,
    )
    return child_chunks


def create_chunk_records(
    text: str,
    filename: str,
    page_number: int,
    logger: logging.Logger | None = None,
) -> dict[str, list[dict[str, str | int | None]]]:
    logger = logger or logging.getLogger("ecm_ai_backend")

    if not text or not text.strip():
        logger.warning("Skipping chunk creation for empty text from %s page %s", filename, page_number)
        return {"parents": [], "children": []}

    parents = create_parent_chunks(text, filename, page_number, logger=logger)
    children = create_child_chunks(parents, filename, page_number, logger=logger)

    return {"parents": parents, "children": children}
