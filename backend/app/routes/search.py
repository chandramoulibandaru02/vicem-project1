from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])
search_service = SearchService()


class SearchMatch(BaseModel):
    chunk: str
    score: float
    metadata: dict
    source_document: str
    filename: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    parent_id: str | None = None


class SourceDocument(BaseModel):
    chunk: str
    metadata: dict
    filename: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    parent_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    filters: dict | None
    matches: list[SearchMatch]
    source_documents: list[SourceDocument]


@router.get(
    "",
    response_model=SearchResponse,
    summary="Run semantic vector search over indexed documents",
    description="Returns matched chunks, similarity scores, metadata, and source documents using ChromaDB semantic retrieval.",
)
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of semantic matches to return"),
    filename: str | None = Query(default=None, description="Optional filename metadata filter"),
    page_number: int | None = Query(default=None, ge=1, description="Optional page metadata filter"),
    request: Request = None,
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")

    retriever = getattr(request.app.state, "retriever", None)

    result = await search_service.search(
        query=q,
        retriever=retriever,
        top_k=top_k,
        filename=filename,
        page_number=page_number,
    )

    return SearchResponse(
        query=result["query"],
        top_k=result["top_k"],
        filters=result["filters"],
        matches=[SearchMatch(**match) for match in result["matches"]],
        source_documents=[SourceDocument(**source) for source in result["source_documents"]],
    )
