"""
Vector API routes — thin handlers delegating to VectorSearchService.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_vector_service
from app.exceptions import VectorServiceError
from app.models.schemas import (
    HybridSearchRequest,
    IndexRequest,
    IndexResponse,
    MetadataSearchRequest,
    SearchResponse,
    SemanticSearchRequest,
    StatsResponse,
    VectorSearchRequest,
)
from app.services.search_service import VectorSearchService

router = APIRouter()


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=201,
    summary="Index a document",
    description="Chunk text, generate embeddings, and store in Chroma with metadata.",
)
def index_document(
    request: IndexRequest,
    service: VectorSearchService = Depends(get_vector_service),
) -> IndexResponse:
    try:
        return service.index_document(request)
    except VectorServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the vector index",
    description=(
        "Unified search endpoint (assignment §7). "
        "Set search_type to semantic, filtered, or hybrid."
    ),
)
def search(
    request: VectorSearchRequest,
    service: VectorSearchService = Depends(get_vector_service),
) -> SearchResponse:
    try:
        return service.unified_search(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type.value,
            filters=request.filters,
            metadata_filters=request.metadata_filters,
            keywords=request.keywords,
        )
    except VectorServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/search/semantic",
    response_model=SearchResponse,
    summary="Semantic (vector-only) search",
)
def search_semantic(
    request: SemanticSearchRequest,
    service: VectorSearchService = Depends(get_vector_service),
) -> SearchResponse:
    try:
        return service.semantic_search(request)
    except VectorServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/search/filtered",
    response_model=SearchResponse,
    summary="Metadata-filtered semantic search",
)
def search_filtered(
    request: MetadataSearchRequest,
    service: VectorSearchService = Depends(get_vector_service),
) -> SearchResponse:
    try:
        return service.metadata_filtered_search(request)
    except VectorServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    summary="Hybrid search (vector + metadata + keywords)",
)
def search_hybrid(
    request: HybridSearchRequest,
    service: VectorSearchService = Depends(get_vector_service),
) -> SearchResponse:
    try:
        return service.hybrid_search(request)
    except VectorServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Vector database statistics",
)
def stats(
    service: VectorSearchService = Depends(get_vector_service),
) -> StatsResponse:
    return service.get_stats()
