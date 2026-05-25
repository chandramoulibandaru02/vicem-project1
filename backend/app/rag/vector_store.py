import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db", logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

    def create_collection(self, collection_name: str = "ecm_documents"):
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(name=collection_name)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str | int | None]],
        ids: list[str],
    ) -> None:
        collection = self.create_collection(collection_name)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        self.logger.info(
            "Added %s documents to collection %s",
            len(ids),
            collection_name,
        )

    def search_documents(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, str | int | bool | None] | None = None,
    ) -> list[dict[str, object]]:
        collection = self.create_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
        )

        formatted = []
        for index in range(len(results["ids"][0])):
            formatted.append(
                {
                    "id": results["ids"][0][index],
                    "document": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        self.logger.info(
            "Search returned %s results for collection %s",
            len(formatted),
            collection_name,
        )
        return formatted


def create_vectorstore(persist_directory: str = "chroma_db", logger: logging.Logger | None = None) -> VectorStore:
    logger = logger or logging.getLogger("ecm_ai_backend")
    logger.info("Creating Chroma vector store at %s", persist_directory)
    return VectorStore(persist_directory=persist_directory, logger=logger)


def add_documents(
    vector_store: VectorStore,
    collection_name: str,
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, str | int | None]],
    ids: list[str],
) -> None:
    vector_store.add_documents(
        collection_name=collection_name,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def search_documents(
    vector_store: VectorStore,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 5,
    filters: dict[str, str | int | bool | None] | None = None,
) -> list[dict[str, object]]:
    return vector_store.search_documents(
        collection_name=collection_name,
        query_embedding=query_embedding,
        top_k=top_k,
        filters=filters,
    )
