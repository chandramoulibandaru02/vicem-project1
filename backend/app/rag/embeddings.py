import logging
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        logger: logging.Logger | None = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:

        self.logger = logger or logging.getLogger(
            "ecm_ai_backend"
        )

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        self.logger.info(
            "Embedding model loaded successfully "
            "(dimension=%s)",
            self.embedding_dimension,
        )

    def generate_embeddings(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:

            self.logger.warning(
                "No valid texts provided for embedding generation"
            )

            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        self.logger.info(
            "Generating embeddings for %s chunks",
            len(cleaned_texts),
        )

        try:

            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return embeddings.astype(np.float32)

        except Exception as e:

            self.logger.exception(
                f"Embedding generation failed: {str(e)}"
            )

            raise

    def generate_chunk_embeddings(
        self,
        chunks: Iterable[
            dict[str, str | int | None]
        ],
    ) -> list[
        dict[
            str,
            str | int | None | np.ndarray,
        ]
    ]:

        chunks = list(chunks)

        texts = [
            str(chunk.get("text", ""))
            for chunk in chunks
        ]

        embeddings = self.generate_embeddings(
            texts
        )

        enriched_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            enriched_chunk = dict(chunk)

            enriched_chunk[
                "embedding"
            ] = embedding

            enriched_chunks.append(
                enriched_chunk
            )

        self.logger.info(
            "Generated embeddings for %s chunks",
            len(enriched_chunks),
        )

        return enriched_chunks

    def generate_single_embedding(
        self,
        text: str,
    ) -> np.ndarray:

        result = self.generate_embeddings(
            [text]
        )

        if len(result) == 0:
            raise ValueError(
                "Failed to generate embedding"
            )

        return result[0]


def generate_embedding_vector(
    text: str,
    generator: EmbeddingGenerator | None = None,
) -> np.ndarray:

    if generator is None:
        generator = EmbeddingGenerator()

    return generator.generate_single_embedding(
        text
    )