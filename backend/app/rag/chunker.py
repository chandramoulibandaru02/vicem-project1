import logging
import uuid
from typing import Iterable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 40


def build_chunk_metadata(
    filename: str,
    document_id: str,
    page_number: int,
    chunk_id: str,
    chunk_index: int,
    parent_id: str | None = None,
    namespace: str = "default",
) -> dict[str, str | int | None]:

    return {
        "filename": filename,
        "document_id": document_id,
        "page_number": page_number,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "parent_id": parent_id,
        "namespace": namespace,
    }


def create_parent_chunks(
    text: str,
    filename: str,
    document_id: str,
    page_number: int,
    namespace: str = "default",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    logger: logging.Logger | None = None,
) -> list[dict]:

    logger = logger or logging.getLogger(
        "ecm_ai_backend"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    parent_texts = splitter.split_text(
        text
    )

    chunks = []

    for index, chunk_text in enumerate(
        parent_texts,
        start=1,
    ):

        chunk_text = chunk_text.strip()

        if (
            len(chunk_text)
            < MIN_CHUNK_LENGTH
        ):
            continue

        chunk_id = (
            f"parent-"
            f"{page_number}-"
            f"{index}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        chunks.append(
            {
                "text": chunk_text,
                "metadata": build_chunk_metadata(
                    filename=filename,
                    document_id=document_id,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    chunk_index=index,
                    namespace=namespace,
                ),
                "level": "parent",
            }
        )

    logger.info(
        "Created %s parent chunks",
        len(chunks),
    )

    return chunks


def create_child_chunks(
    parent_chunks: Iterable[dict],
    filename: str,
    document_id: str,
    page_number: int,
    namespace: str = "default",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    logger: logging.Logger | None = None,
) -> list[dict]:

    logger = logger or logging.getLogger(
        "ecm_ai_backend"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    child_chunks = []

    for parent_chunk in parent_chunks:

        parent_id = str(
            parent_chunk["metadata"][
                "chunk_id"
            ]
        )

        parent_text = str(
            parent_chunk["text"]
        )

        child_texts = splitter.split_text(
            parent_text
        )

        for index, child_text in enumerate(
            child_texts,
            start=1,
        ):

            child_text = child_text.strip()

            if (
                len(child_text)
                < MIN_CHUNK_LENGTH
            ):
                continue

            child_id = (
                f"child-"
                f"{page_number}-"
                f"{index}-"
                f"{uuid.uuid4().hex[:8]}"
            )

            child_chunks.append(
                {
                    "text": child_text,
                    "metadata": build_chunk_metadata(
                        filename=filename,
                        document_id=document_id,
                        page_number=page_number,
                        chunk_id=child_id,
                        chunk_index=index,
                        parent_id=parent_id,
                        namespace=namespace,
                    ),
                    "level": "child",
                }
            )

    logger.info(
        "Created %s child chunks",
        len(child_chunks),
    )

    return child_chunks


def create_chunk_records(
    text: str,
    filename: str,
    document_id: str,
    page_number: int,
    namespace: str = "default",
    logger: logging.Logger | None = None,
) -> dict[str, list[dict]]:

    logger = logger or logging.getLogger(
        "ecm_ai_backend"
    )

    if (
        not text
        or not text.strip()
    ):

        logger.warning(
            "Skipping empty page"
        )

        return {
            "parents": [],
            "children": [],
        }

    parents = create_parent_chunks(
        text=text,
        filename=filename,
        document_id=document_id,
        page_number=page_number,
        namespace=namespace,
        logger=logger,
    )

    children = create_child_chunks(
        parent_chunks=parents,
        filename=filename,
        document_id=document_id,
        page_number=page_number,
        namespace=namespace,
        logger=logger,
    )

    return {
        "parents": parents,
        "children": children,
    }