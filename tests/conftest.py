"""Pytest fixtures for API tests."""

import os
import shutil
import tempfile
from typing import Generator, List

import pytest
from fastapi.testclient import TestClient

from app.dependencies import reset_vector_service
from app.main import app
from app.repositories.chroma_repository import ChromaRepository
from app.services.chunking_service import ChunkingService
from app.services.search_service import VectorSearchService


class MockEmbeddingService:
    """Lightweight embedder for tests (avoids loading sentence-transformers)."""

    dimension = 384

    def encode(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        return [0.1] * 384

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [[0.1] * 384 for _ in texts]

    def get_dimension(self) -> int:
        return 384


@pytest.fixture()
def chroma_temp_dir() -> Generator[str, None, None]:
    path = tempfile.mkdtemp(prefix="chroma_test_")
    os.environ["CHROMA_PERSIST_DIRECTORY"] = path
    yield path
    shutil.rmtree(path, ignore_errors=True)
    reset_vector_service()


@pytest.fixture()
def vector_client(chroma_temp_dir: str) -> Generator[TestClient, None, None]:
    reset_vector_service()
    repo = ChromaRepository(
        persist_directory=chroma_temp_dir,
        collection_name="test_api_collection",
    )
    service = VectorSearchService(
        repository=repo,
        embedder=MockEmbeddingService(),
        chunker=ChunkingService(chunk_size=100, chunk_overlap=20),
    )

    import app.dependencies as deps

    deps._service = service

    with TestClient(app) as client:
        yield client

    reset_vector_service()


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "source": "ocr",
        "page_number": 1,
        "created_at": "2026-01-14T10:00:00Z",
        "tags": ["medical", "invoice"],
    }
