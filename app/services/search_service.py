"""
VectorSearchService: orchestrates chunking, embedding, and Chroma retrieval.

API routes delegate here; no vector logic in route handlers (assignment §8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import settings
from app.exceptions import (
    DimensionMismatchError,
    IndexEmptyError,
    InvalidFiltersError,
    InvalidSearchTypeError,
    VectorServiceError,
)
from app.models.schemas import (
    HybridSearchRequest,
    IndexRequest,
    IndexResponse,
    MetadataSearchRequest,
    SearchResponse,
    SearchResult,
    SemanticSearchRequest,
    StatsResponse,
)
from app.repositories.chroma_repository import ChromaRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService


class VectorSearchService:
    """Business logic for indexing and search."""

    def __init__(
        self,
        repository: Optional[ChromaRepository] = None,
        embedder: Optional[EmbeddingService] = None,
        chunker: Optional[ChunkingService] = None,
    ) -> None:
        self.repository = repository or ChromaRepository()
        self.embedder = embedder or EmbeddingService()
        self.chunker = chunker or ChunkingService()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(self, request: IndexRequest) -> IndexResponse:
        """Chunk content, embed, and persist with metadata."""
        self._validate_document_metadata(request.metadata)

        base_metadata = {
            **request.metadata,
            "document_id": request.document_id,
            "user_id": request.user_id,
        }

        chunks_with_meta = self.chunker.chunk_with_metadata(
            request.content,
            base_metadata,
        )
        if not chunks_with_meta:
            raise VectorServiceError("No chunks produced from document content")

        texts = [item["text"] for item in chunks_with_meta]
        embeddings = self.embedder.encode_batch(texts)
        self._assert_embedding_dimensions(embeddings)

        metadatas = [
            self._chroma_metadata(item["metadata"], request.user_id, request.document_id)
            for item in chunks_with_meta
        ]

        chunk_ids = self.repository.add_chunks(
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id=request.document_id,
        )

        return IndexResponse(
            message="Document indexed successfully",
            document_id=request.document_id,
            chunks_created=len(chunk_ids),
            chunk_ids=chunk_ids,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def semantic_search(self, request: SemanticSearchRequest) -> SearchResponse:
        self._ensure_index_ready()
        query_embedding = self._embed_query(request.query)
        raw = self.repository.semantic_search(
            query_embedding=query_embedding,
            n_results=min(request.top_k, settings.MAX_TOP_K),
        )
        return self._to_search_response(raw, request.query)

    def metadata_filtered_search(
        self, request: MetadataSearchRequest
    ) -> SearchResponse:
        self._ensure_index_ready()
        if not request.filters:
            raise InvalidFiltersError("filters are required for metadata-filtered search")

        query_embedding = self._embed_query(request.query)
        raw = self.repository.metadata_filtered_search(
            query_embedding=query_embedding,
            filters=request.filters,
            n_results=min(request.top_k, settings.MAX_TOP_K),
        )
        return self._to_search_response(raw, request.query)

    def hybrid_search(self, request: HybridSearchRequest) -> SearchResponse:
        self._ensure_index_ready()
        query_embedding = self._embed_query(request.query)
        where_document = self._keywords_to_where_document(request.keywords)

        raw = self.repository.hybrid_search(
            query_embedding=query_embedding,
            metadata_filters=request.metadata_filters,
            where_document=where_document,
            n_results=min(request.top_k, settings.MAX_TOP_K),
        )
        return self._to_search_response(raw, request.query)

    def unified_search(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "semantic",
        filters: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None,
    ) -> SearchResponse:
        """Single entry point for POST /vector/search (assignment §7)."""
        normalized = search_type.strip().lower()
        if normalized == "semantic":
            return self.semantic_search(
                SemanticSearchRequest(query=query, top_k=top_k)
            )
        if normalized == "filtered":
            return self.metadata_filtered_search(
                MetadataSearchRequest(
                    query=query,
                    filters=filters or {},
                    top_k=top_k,
                )
            )
        if normalized == "hybrid":
            return self.hybrid_search(
                HybridSearchRequest(
                    query=query,
                    metadata_filters=metadata_filters,
                    keywords=keywords,
                    top_k=top_k,
                )
            )
        raise InvalidSearchTypeError(search_type)

    def get_stats(self) -> StatsResponse:
        stats = self.repository.get_stats()
        chunk_stats = self.chunker.get_stats()
        return StatsResponse(
            total_chunks=stats["total_chunks"],
            collection_name=stats["collection_name"],
            embedding_dimension=stats.get("embedding_dimension"),
            persist_directory=stats.get("persist_directory"),
            chunk_size=chunk_stats["chunk_size"],
            chunk_overlap=chunk_stats["chunk_overlap"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> List[float]:
        embedding = self.embedder.encode(query)
        if len(embedding) != settings.EMBEDDING_DIMENSION:
            raise DimensionMismatchError(
                settings.EMBEDDING_DIMENSION,
                len(embedding),
            )
        return embedding

    def _assert_embedding_dimensions(self, embeddings: List[List[float]]) -> None:
        for emb in embeddings:
            if len(emb) != settings.EMBEDDING_DIMENSION:
                raise DimensionMismatchError(
                    settings.EMBEDDING_DIMENSION,
                    len(emb),
                )

    def _ensure_index_ready(self) -> None:
        if self.repository.get_stats()["total_chunks"] == 0:
            raise IndexEmptyError()

    @staticmethod
    def _validate_document_metadata(metadata: Dict[str, Any]) -> None:
        required = ("source", "page_number", "created_at")
        missing = [k for k in required if k not in metadata]
        if missing:
            raise InvalidFiltersError(
                f"metadata missing required fields: {', '.join(missing)}"
            )
        allowed_sources = ("ocr", "pdf", "image")
        if metadata["source"] not in allowed_sources:
            raise InvalidFiltersError(
                f"metadata.source must be one of {allowed_sources}"
            )

    @staticmethod
    def _chroma_metadata(
        chunk_meta: Dict[str, Any],
        user_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Flatten metadata for Chroma (str/int/float/bool only)."""
        meta = {
            **chunk_meta,
            "user_id": str(user_id),
            "document_id": str(document_id),
        }
        tags = meta.get("tags")
        if isinstance(tags, list):
            meta["tags"] = ",".join(str(t) for t in tags)
        for key, value in list(meta.items()):
            if value is None:
                del meta[key]
            elif isinstance(value, (dict, list)):
                meta[key] = str(value)
        return meta

    @staticmethod
    def _keywords_to_where_document(
        keywords: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        if not keywords:
            return None
        cleaned = [k.strip() for k in keywords if k and k.strip()]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return {"$contains": cleaned[0]}
        return {
            "$and": [{"$contains": kw} for kw in cleaned],
        }

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Chroma cosine distance → similarity in [0, 1]."""
        return max(0.0, min(1.0, 1.0 - float(distance)))

    def _to_search_response(
        self,
        raw: Dict[str, Any],
        query: str,
    ) -> SearchResponse:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: List[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if chunk_id is None:
                continue
            dist = float(distance) if distance is not None else 1.0
            results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=content or "",
                    metadata=metadata or {},
                    distance=dist,
                    similarity_score=self._distance_to_similarity(dist),
                )
            )

        return SearchResponse(results=results, count=len(results), query=query)
