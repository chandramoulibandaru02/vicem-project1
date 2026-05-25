import logging
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        logger: logging.Logger | None = None,
        batch_size: int = 32,
    ) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")

    def generate_embeddings(self, texts: Iterable[str]) -> np.ndarray:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            self.logger.warning("No embeddings generated because the text list is empty")
            return np.empty((0, 0), dtype=np.float32)

        self.logger.info("Generating embeddings for %s chunks", len(cleaned_texts))

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def generate_chunk_embeddings(self, chunks: Iterable[dict[str, str | int | None]]) -> list[dict[str, str | int | None | np.ndarray]]:
        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunk = dict(chunk)
            enriched_chunk["embedding"] = embedding
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks


def generate_embedding_vector(text: str, generator: EmbeddingGenerator | None = None) -> np.ndarray:
    if generator is None:
        generator = EmbeddingGenerator()

    result = generator.generate_embeddings([text])
    return result
