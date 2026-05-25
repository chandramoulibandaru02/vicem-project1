import logging

from fastapi import HTTPException


class SearchService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend.search")

    def _build_filters(self, filename: str | None, page_number: int | None) -> dict | None:
        filters: dict[str, str | int] = {}
        if filename:
            filters["filename"] = filename
        if page_number is not None:
            filters["page_number"] = page_number
        return filters or None

    def _serialize_match(self, document, score: float) -> dict[str, object]:
        metadata = dict(document.metadata or {})
        return {
            "chunk": document.page_content,
            "score": round(float(score), 6),
            "metadata": metadata,
            "source_document": document.page_content,
            "filename": metadata.get("filename"),
            "page_number": metadata.get("page_number"),
            "chunk_id": metadata.get("chunk_id"),
            "parent_id": metadata.get("parent_id"),
        }

    async def search(self, query: str, retriever, top_k: int = 5, filename: str | None = None, page_number: int | None = None) -> dict[str, object]:
        if retriever is None:
            raise HTTPException(status_code=503, detail="RAG retriever is not initialized")

        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="q is required")

        filters = self._build_filters(filename, page_number)

        try:
            self.logger.info(
                "Semantic search request",
                extra={"query": query[:120], "top_k": top_k, "filters": filters},
            )
            results = await retriever.search(query, top_k=top_k, filters=filters)
            matches = [self._serialize_match(document, score) for document, score in results]
            source_documents = [
                {
                    "chunk": document.page_content,
                    "metadata": dict(document.metadata or {}),
                    "filename": (document.metadata or {}).get("filename"),
                    "page_number": (document.metadata or {}).get("page_number"),
                    "chunk_id": (document.metadata or {}).get("chunk_id"),
                    "parent_id": (document.metadata or {}).get("parent_id"),
                }
                for document, _ in results
            ]

            self.logger.info("Semantic search completed", extra={"match_count": len(matches)})
            return {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "matches": matches,
                "source_documents": source_documents,
            }
        except HTTPException:
            raise
        except RuntimeError as exc:
            self.logger.exception("Runtime error while running semantic search")
            raise HTTPException(status_code=503, detail="Semantic search is temporarily unavailable") from exc
        except Exception as exc:
            self.logger.exception("Unexpected error while running semantic search")
            raise HTTPException(status_code=500, detail="Failed to complete semantic search") from exc
