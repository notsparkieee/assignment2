"""FastAPI dependencies and shared service instances."""

from __future__ import annotations

from typing import Optional

from app.services.search_service import VectorSearchService

_service: Optional[VectorSearchService] = None


def get_vector_service() -> VectorSearchService:
    """Return a process-wide VectorSearchService (lazy singleton)."""
    global _service
    if _service is None:
        _service = VectorSearchService()
    return _service


def reset_vector_service() -> None:
    """Clear singleton (used in tests)."""
    global _service
    _service = None
