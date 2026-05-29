"""Phase 3: VectorSearchService unit/integration tests (no HTTP layer)."""

import os
import shutil
import tempfile

import pytest

from app.repositories.chroma_repository import ChromaRepository
from app.services.chunking_service import ChunkingService
from app.services.search_service import VectorSearchService
from app.models.schemas import (
    HybridSearchRequest,
    IndexRequest,
    MetadataSearchRequest,
    SemanticSearchRequest,
)


class MockEmbeddingService:
    dimension = 384

    def encode(self, text: str):
        return [0.1] * 384

    def encode_batch(self, texts):
        return [[0.1] * 384 for _ in texts]

    def get_dimension(self):
        return 384


@pytest.fixture()
def service() -> VectorSearchService:
    path = tempfile.mkdtemp(prefix="chroma_phase3_")
    os.environ["CHROMA_PERSIST_DIRECTORY"] = path
    repo = ChromaRepository(
        persist_directory=path,
        collection_name="phase3_test_collection",
    )
    svc = VectorSearchService(
        repository=repo,
        embedder=MockEmbeddingService(),
        chunker=ChunkingService(chunk_size=80, chunk_overlap=10),
    )
    yield svc
    shutil.rmtree(path, ignore_errors=True)


def _index_sample(svc: VectorSearchService) -> None:
    svc.index_document(
        IndexRequest(
            document_id="doc_p3",
            user_id="user_1",
            content="Diabetes insulin therapy and blood sugar monitoring for patients.",
            metadata={
                "source": "ocr",
                "page_number": 1,
                "created_at": "2026-01-14T10:00:00Z",
                "tags": ["medical"],
            },
        )
    )


def test_index_and_semantic_search(service: VectorSearchService):
    _index_sample(service)
    result = service.semantic_search(
        SemanticSearchRequest(query="insulin blood sugar", top_k=3)
    )
    assert result.count >= 1
    assert result.results[0].similarity_score >= 0


def test_filtered_search(service: VectorSearchService):
    _index_sample(service)
    result = service.metadata_filtered_search(
        MetadataSearchRequest(
            query="medical treatment",
            filters={"source": "ocr"},
            top_k=3,
        )
    )
    assert result.count >= 1


def test_hybrid_search(service: VectorSearchService):
    _index_sample(service)
    result = service.hybrid_search(
        HybridSearchRequest(
            query="diabetes",
            metadata_filters={"source": "ocr"},
            keywords=["insulin"],
            top_k=3,
        )
    )
    assert result.count >= 1


def test_empty_index_raises(service: VectorSearchService):
    from app.exceptions import IndexEmptyError

    with pytest.raises(IndexEmptyError):
        service.semantic_search(SemanticSearchRequest(query="test", top_k=3))
