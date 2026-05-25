import logging
import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


class VectorStore:
    def __init__(
        self,
        index_name: str | None = None,
        namespace: str = "default",
        logger: logging.Logger | None = None,
    ) -> None:

        self.logger = logger or logging.getLogger(
            "ecm_ai_backend"
        )

        self.api_key = os.getenv("PINECONE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY not found"
            )

        self.index_name = (
            index_name
            or os.getenv("PINECONE_INDEX_NAME")
            or "ecm-documents"
        )

        self.namespace = namespace

        try:
            # Initialize Pinecone
            self.pc = Pinecone(
                api_key=self.api_key
            )

            # Connect to index
            self.index = self.pc.Index(
                self.index_name
            )

            self.logger.info(
                "Connected to Pinecone index: %s",
                self.index_name,
            )

        except Exception as e:
            self.logger.exception(
                f"Pinecone initialization failed: {str(e)}"
            )
            raise

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[
            dict[str, str | int | float | bool | None]
        ],
        ids: list[str],
    ) -> None:

        if not (
            len(documents)
            == len(embeddings)
            == len(metadatas)
            == len(ids)
        ):
            raise ValueError(
                "documents, embeddings, metadatas, ids length mismatch"
            )

        vectors = []

        for i in range(len(ids)):

            metadata = metadatas[i] or {}

            metadata["text"] = documents[i]

            vectors.append(
                {
                    "id": ids[i],
                    "values": embeddings[i],
                    "metadata": metadata,
                }
            )

        try:
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace,
            )

            self.logger.info(
                "Inserted %s vectors into Pinecone",
                len(vectors),
            )

        except Exception as e:
            self.logger.exception(
                f"Vector upsert failed: {str(e)}"
            )
            raise

    def search_documents(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, object]]:

        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filters,
            )

            formatted_results = []

            for match in response.matches:

                metadata = match.metadata or {}

                formatted_results.append(
                    {
                        "id": match.id,
                        "document": metadata.get(
                            "text",
                            "",
                        ),
                        "metadata": metadata,
                        "score": match.score,
                    }
                )

            self.logger.info(
                "Retrieved %s results from Pinecone",
                len(formatted_results),
            )

            return formatted_results

        except Exception as e:
            self.logger.exception(
                f"Vector search failed: {str(e)}"
            )
            raise

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:

        try:
            self.index.delete(
                ids=ids,
                namespace=self.namespace,
            )

            self.logger.info(
                "Deleted %s vectors",
                len(ids),
            )

        except Exception as e:
            self.logger.exception(
                f"Vector deletion failed: {str(e)}"
            )
            raise

    def clear_namespace(self) -> None:

        try:
            self.index.delete(
                delete_all=True,
                namespace=self.namespace,
            )

            self.logger.info(
                "Cleared namespace: %s",
                self.namespace,
            )

        except Exception as e:
            self.logger.exception(
                f"Namespace cleanup failed: {str(e)}"
            )
            raise


def create_vectorstore(
    logger: logging.Logger | None = None,
    namespace: str = "default",
) -> VectorStore:

    logger = logger or logging.getLogger(
        "ecm_ai_backend"
    )

    logger.info(
        "Creating Pinecone vector store"
    )

    return VectorStore(
        logger=logger,
        namespace=namespace,
    )


def add_documents(
    vector_store: VectorStore,
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[
        dict[str, str | int | float | bool | None]
    ],
    ids: list[str],
) -> None:

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def search_documents(
    vector_store: VectorStore,
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, object]]:

    return vector_store.search_documents(
        query_embedding=query_embedding,
        top_k=top_k,
        filters=filters,
    )