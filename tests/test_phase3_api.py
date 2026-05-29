"""Phase 3–4: FastAPI vector endpoints."""

import pytest


INDEX_PAYLOAD = {
    "document_id": "doc_api_001",
    "user_id": "user_001",
    "content": (
        "Diabetes mellitus requires insulin therapy and blood sugar monitoring. "
        "Patients should follow a medical diet and exercise plan."
    ),
    "metadata": {
        "source": "ocr",
        "page_number": 1,
        "created_at": "2026-01-14T10:00:00Z",
        "tags": ["medical", "diabetes"],
    },
}


def test_health(vector_client):
    response = vector_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_index_and_stats(vector_client):
    response = vector_client.post("/vector/index", json=INDEX_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc_api_001"
    assert body["chunks_created"] >= 1

    stats = vector_client.get("/vector/stats")
    assert stats.status_code == 200
    data = stats.json()
    assert data["total_chunks"] >= 1
    assert data["embedding_dimension"] == 384


def test_semantic_search_unified(vector_client):
    vector_client.post("/vector/index", json=INDEX_PAYLOAD)
    response = vector_client.post(
        "/vector/search",
        json={
            "query": "insulin and blood sugar",
            "top_k": 5,
            "search_type": "semantic",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert data["results"][0]["similarity_score"] >= 0


def test_filtered_search(vector_client):
    vector_client.post("/vector/index", json=INDEX_PAYLOAD)
    response = vector_client.post(
        "/vector/search/filtered",
        json={
            "query": "medical treatment",
            "filters": {"source": "ocr"},
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] >= 1


def test_hybrid_search(vector_client):
    vector_client.post("/vector/index", json=INDEX_PAYLOAD)
    response = vector_client.post(
        "/vector/search/hybrid",
        json={
            "query": "diabetes monitoring",
            "metadata_filters": {"source": "ocr"},
            "keywords": ["insulin"],
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] >= 1


def test_search_empty_index_returns_404(vector_client):
    response = vector_client.post(
        "/vector/search",
        json={"query": "test", "search_type": "semantic"},
    )
    assert response.status_code == 404


def test_index_missing_metadata_returns_400(vector_client):
    bad = {**INDEX_PAYLOAD, "metadata": {"source": "ocr"}}
    response = vector_client.post("/vector/index", json=bad)
    assert response.status_code == 400


def test_empty_query_returns_422(vector_client):
    response = vector_client.post(
        "/vector/search",
        json={"query": "   ", "search_type": "semantic"},
    )
    assert response.status_code == 422


def test_invalid_search_type(vector_client):
    vector_client.post("/vector/index", json=INDEX_PAYLOAD)
    response = vector_client.post(
        "/vector/search",
        json={
            "query": "diabetes",
            "search_type": "invalid_mode",
        },
    )
    assert response.status_code == 422
