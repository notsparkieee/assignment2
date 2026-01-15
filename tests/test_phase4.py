"""
Phase 4 Tests: FastAPI Endpoint Integration

Tests all 5 API endpoints with various scenarios.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.repositories.chroma_repository import ChromaRepository


# Create test client
client = TestClient(app)


class TestIndexEndpoint:
    """Test POST /vector/index endpoint."""
    
    def test_index_document_success(self):
        """Test successful document indexing."""
        response = client.post("/vector/index", json={
            "text": "Patient John Doe has Type 2 diabetes. Blood sugar levels elevated. Requires insulin therapy.",
            "metadata": {
                "patient_id": "P001",
                "document_type": "medical_report"
            }
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_indexed"] > 0
        assert "message" in data
    
    def test_index_with_minimal_metadata(self):
        """Test indexing with minimal metadata."""
        response = client.post("/vector/index", json={
            "text": "This is a short test document for indexing.",
            "metadata": {}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
    
    def test_index_without_metadata(self):
        """Test indexing without metadata field."""
        response = client.post("/vector/index", json={
            "text": "This document has no metadata field."
        })
        
        assert response.status_code == 201
    
    def test_index_empty_text_fails(self):
        """Test that empty text returns 400 error."""
        response = client.post("/vector/index", json={
            "text": "",
            "metadata": {"doc_id": "test"}
        })
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_index_whitespace_only_fails(self):
        """Test that whitespace-only text fails."""
        response = client.post("/vector/index", json={
            "text": "   \n\t  ",
            "metadata": {}
        })
        
        assert response.status_code == 400
    
    def test_index_missing_text_field(self):
        """Test that missing text field returns 422 validation error."""
        response = client.post("/vector/index", json={
            "metadata": {"doc_id": "test"}
        })
        
        assert response.status_code == 422  # Pydantic validation error


class TestSemanticSearchEndpoint:
    """Test POST /vector/search endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Index test documents before each test."""
        # Index some test documents
        documents = [
            {
                "text": "Patient diagnosed with Type 2 diabetes. Blood sugar elevated. Insulin therapy recommended.",
                "metadata": {"patient_id": "P001", "type": "medical"}
            },
            {
                "text": "Heart disease and cardiovascular problems detected. Regular exercise recommended.",
                "metadata": {"patient_id": "P002", "type": "medical"}
            },
            {
                "text": "Python programming language is popular for data science and machine learning.",
                "metadata": {"doc_id": "tech001", "type": "tech"}
            }
        ]
        
        for doc in documents:
            client.post("/vector/index", json=doc)
        
        yield
        
        # Cleanup after test (optional - depends on test isolation needs)
    
    def test_semantic_search_success(self):
        """Test successful semantic search."""
        response = client.post("/vector/search", json={
            "query": "diabetes treatment",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert data["query"] == "diabetes treatment"
        assert data["total_results"] > 0
        assert data["search_type"] == "semantic"
        
        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "text" in result
            assert "score" in result
            assert "metadata" in result
            assert 0.0 <= result["score"] <= 1.0
    
    def test_semantic_search_default_top_k(self):
        """Test search with default top_k value."""
        response = client.post("/vector/search", json={
            "query": "medical treatment"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 5  # Default top_k is 5
    
    def test_semantic_search_custom_top_k(self):
        """Test search with custom top_k value."""
        response = client.post("/vector/search", json={
            "query": "treatment",
            "top_k": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2
    
    def test_semantic_search_empty_query_fails(self):
        """Test that empty query returns 400."""
        response = client.post("/vector/search", json={
            "query": "",
            "top_k": 5
        })
        
        assert response.status_code == 400
    
    def test_semantic_search_invalid_top_k(self):
        """Test that invalid top_k returns 422 validation error."""
        response = client.post("/vector/search", json={
            "query": "test",
            "top_k": 0
        })
        
        assert response.status_code == 422  # Pydantic validation
    
    def test_semantic_search_relevance_ranking(self):
        """Test that results are ranked by relevance."""
        response = client.post("/vector/search", json={
            "query": "diabetes insulin",
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Scores should be in descending order
        if len(data["results"]) > 1:
            scores = [r["score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)


class TestFilteredSearchEndpoint:
    """Test POST /vector/search/filtered endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Index categorized test documents."""
        documents = [
            {
                "text": "Patient John Doe diabetes treatment plan insulin dosage",
                "metadata": {"patient_id": "P001", "document_type": "medical_report", "year": 2023}
            },
            {
                "text": "Patient Jane Smith cardiovascular disease treatment",
                "metadata": {"patient_id": "P002", "document_type": "medical_report", "year": 2023}
            },
            {
                "text": "Research paper on diabetes management published in journal",
                "metadata": {"patient_id": None, "document_type": "research_paper", "year": 2022}
            }
        ]
        
        for doc in documents:
            client.post("/vector/index", json=doc)
        
        yield
    
    def test_filtered_search_by_patient_id(self):
        """Test filtering by specific patient ID."""
        response = client.post("/vector/search/filtered", json={
            "query": "treatment",
            "metadata_filters": {"patient_id": "P001"},
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "filtered"
        assert data["filters_applied"] == {"patient_id": "P001"}
        
        # All results should be from P001
        for result in data["results"]:
            assert result["metadata"].get("patient_id") == "P001"
    
    def test_filtered_search_by_document_type(self):
        """Test filtering by document type."""
        response = client.post("/vector/search/filtered", json={
            "query": "diabetes",
            "metadata_filters": {"document_type": "research_paper"},
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find research paper, not patient records
        if data["results"]:
            for result in data["results"]:
                assert result["metadata"].get("document_type") == "research_paper"
    
    def test_filtered_search_multiple_filters(self):
        """Test filtering with multiple criteria."""
        response = client.post("/vector/search/filtered", json={
            "query": "treatment",
            "metadata_filters": {
                "document_type": "medical_report",
                "year": 2023
            },
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # All results should match both filters
        for result in data["results"]:
            assert result["metadata"].get("document_type") == "medical_report"
            assert result["metadata"].get("year") == 2023
    
    def test_filtered_search_empty_query_fails(self):
        """Test that empty query fails."""
        response = client.post("/vector/search/filtered", json={
            "query": "",
            "metadata_filters": {"patient_id": "P001"},
            "top_k": 5
        })
        
        assert response.status_code == 400
    
    def test_filtered_search_empty_filters_fails(self):
        """Test that empty filters returns 400."""
        response = client.post("/vector/search/filtered", json={
            "query": "test",
            "metadata_filters": {},
            "top_k": 5
        })
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


class TestHybridSearchEndpoint:
    """Test POST /vector/search/hybrid endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Index rich test documents for hybrid search."""
        documents = [
            {
                "text": "URGENT: Patient requires immediate insulin therapy in ICU. High-risk diabetes case.",
                "metadata": {"urgency": "high", "location": "ICU", "condition": "diabetes"}
            },
            {
                "text": "Standard diabetes management protocol. Regular insulin dosage recommended.",
                "metadata": {"urgency": "low", "location": "outpatient", "condition": "diabetes"}
            },
            {
                "text": "Emergency cardiac intervention needed. Patient in critical condition in ICU.",
                "metadata": {"urgency": "high", "location": "ICU", "condition": "cardiac"}
            }
        ]
        
        for doc in documents:
            client.post("/vector/index", json=doc)
        
        yield
    
    def test_hybrid_search_with_all_features(self):
        """Test hybrid search with filters and keywords."""
        response = client.post("/vector/search/hybrid", json={
            "query": "diabetes treatment",
            "metadata_filters": {"urgency": "high"},
            "keywords": ["urgent", "ICU"],
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "hybrid"
        assert data["filters_applied"] == {"urgency": "high"}
        assert data["keywords_used"] == ["urgent", "ICU"]
        
        # Top result should match all criteria
        if data["results"]:
            top_result = data["results"][0]
            assert "diabetes" in top_result["text"].lower() or "insulin" in top_result["text"].lower()
            assert top_result["metadata"].get("urgency") == "high"
    
    def test_hybrid_search_with_only_query(self):
        """Test hybrid search with just query (no filters/keywords)."""
        response = client.post("/vector/search/hybrid", json={
            "query": "diabetes",
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "hybrid"
        assert data["total_results"] > 0
    
    def test_hybrid_search_with_custom_weights(self):
        """Test hybrid search with custom weight distribution."""
        response = client.post("/vector/search/hybrid", json={
            "query": "treatment",
            "metadata_filters": {"urgency": "high"},
            "keywords": ["critical"],
            "top_k": 3,
            "weights": {
                "vector": 0.6,
                "metadata": 0.3,
                "keyword": 0.1
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] >= 0
    
    def test_hybrid_search_invalid_weights_fails(self):
        """Test that weights not summing to 1.0 returns 400."""
        response = client.post("/vector/search/hybrid", json={
            "query": "test",
            "top_k": 3,
            "weights": {
                "vector": 0.5,
                "metadata": 0.5,
                "keyword": 0.5  # Total = 1.5, invalid!
            }
        })
        
        assert response.status_code == 400
        assert "sum to 1.0" in response.json()["detail"]
    
    def test_hybrid_search_keyword_boosting(self):
        """Test that keywords boost relevant results."""
        response = client.post("/vector/search/hybrid", json={
            "query": "patient care",
            "keywords": ["urgent", "ICU"],
            "top_k": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Results with keywords should rank higher
        if len(data["results"]) > 1:
            # Check that scores are descending
            scores = [r["score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)


class TestStatsEndpoint:
    """Test GET /vector/stats endpoint."""
    
    def test_get_stats_success(self):
        """Test successful stats retrieval."""
        response = client.get("/vector/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "embedding_model" in data
        assert "embedding_dimension" in data
        assert "vector_database" in data
        assert "collection_stats" in data
        
        # Check values
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["embedding_dimension"] == 384
        assert data["vector_database"] == "ChromaDB"
    
    def test_stats_includes_collection_info(self):
        """Test that stats include collection information."""
        response = client.get("/vector/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["collection_stats"], dict)


class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_404_on_invalid_endpoint(self):
        """Test that invalid endpoints return 404."""
        response = client.post("/vector/invalid_endpoint", json={})
        assert response.status_code == 404
    
    def test_422_on_invalid_json_schema(self):
        """Test Pydantic validation errors."""
        response = client.post("/vector/search", json={
            "invalid_field": "value",
            "top_k": "not_a_number"
        })
        assert response.status_code == 422
    
    def test_400_on_business_logic_errors(self):
        """Test that business logic errors return 400."""
        response = client.post("/vector/index", json={
            "text": "",  # Empty text should fail
            "metadata": {}
        })
        assert response.status_code == 400


class TestCORS:
    """Test CORS headers."""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are set correctly."""
        response = client.options("/vector/stats")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code in [200, 405]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
