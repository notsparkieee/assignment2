"""
Phase 3 Tests: SearchService - Semantic, Filtered, and Hybrid Search

Tests all three search strategies and helper functions.
"""

import pytest
from app.services.search_service import SearchService
from app.services.embedding_service import EmbeddingService
from app.services.chunking_service import ChunkingService
from app.repositories.chroma_repository import ChromaRepository
from app.models.schemas import SearchResult


class TestSearchServiceInitialization:
    """Test SearchService initialization and dependency injection."""
    
    def test_default_initialization(self):
        """Test that SearchService initializes with default dependencies."""
        search = SearchService()
        
        assert search.embedding_service is not None
        assert search.chroma_repository is not None
        assert isinstance(search.embedding_service, EmbeddingService)
        assert isinstance(search.chroma_repository, ChromaRepository)
    
    def test_custom_dependencies(self):
        """Test dependency injection with custom services."""
        custom_embedder = EmbeddingService()
        custom_repo = ChromaRepository()
        
        search = SearchService(
            embedding_service=custom_embedder,
            chroma_repository=custom_repo
        )
        
        assert search.embedding_service is custom_embedder
        assert search.chroma_repository is custom_repo


class TestSemanticSearch:
    """Test pure vector similarity search."""
    
    @pytest.fixture
    def search_service(self):
        """Create SearchService with fresh database."""
        # Create services
        chunker = ChunkingService(chunk_size=100, chunk_overlap=20)
        embedder = EmbeddingService()
        repo = ChromaRepository(collection_name="test_semantic_search")
        search = SearchService(embedding_service=embedder, chroma_repository=repo)
        
        # Add test documents
        test_docs = [
            {
                "text": "Patient diagnosed with Type 2 diabetes. Blood sugar levels elevated. Requires insulin therapy.",
                "metadata": {"doc_id": "doc1", "type": "medical"}
            },
            {
                "text": "Heart disease and cardiovascular problems are common. Regular exercise recommended.",
                "metadata": {"doc_id": "doc2", "type": "medical"}
            },
            {
                "text": "Python programming language is popular for data science and machine learning applications.",
                "metadata": {"doc_id": "doc3", "type": "tech"}
            }
        ]
        
        for doc in test_docs:
            chunks = chunker.chunk_text(doc["text"])
            embeddings = embedder.encode_batch(chunks)
            repo.add_chunks(chunks, embeddings, [doc["metadata"]] * len(chunks))
        
        yield search
        
        # Cleanup
        repo.delete_collection()
    
    def test_semantic_search_finds_relevant_results(self, search_service):
        """Test that semantic search returns relevant results."""
        results = search_service.semantic_search("diabetes treatment", top_k=2)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        
        # First result should be about diabetes
        assert "diabetes" in results[0].text.lower() or "insulin" in results[0].text.lower()
    
    def test_semantic_search_returns_scores(self, search_service):
        """Test that results include similarity scores."""
        results = search_service.semantic_search("heart problems", top_k=2)
        
        assert len(results) > 0
        for result in results:
            assert 0.0 <= result.score <= 1.0
            assert result.metadata is not None
    
    def test_semantic_search_respects_top_k(self, search_service):
        """Test that top_k parameter limits results."""
        results_1 = search_service.semantic_search("medical", top_k=1)
        results_2 = search_service.semantic_search("medical", top_k=2)
        
        assert len(results_1) == 1
        assert len(results_2) == 2
    
    def test_semantic_search_empty_query_raises_error(self, search_service):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.semantic_search("", top_k=5)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.semantic_search("   ", top_k=5)
    
    def test_semantic_search_invalid_top_k_raises_error(self, search_service):
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            search_service.semantic_search("test", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            search_service.semantic_search("test", top_k=-1)


class TestMetadataFilteredSearch:
    """Test metadata-filtered search."""
    
    @pytest.fixture
    def search_service(self):
        """Create SearchService with categorized test data."""
        chunker = ChunkingService(chunk_size=100, chunk_overlap=20)
        embedder = EmbeddingService()
        repo = ChromaRepository(collection_name="test_filtered_search")
        search = SearchService(embedding_service=embedder, chroma_repository=repo)
        
        # Add test documents with different metadata
        test_docs = [
            {
                "text": "Patient John Doe requires insulin treatment for diabetes.",
                "metadata": {"patient_id": "P001", "type": "medical", "year": 2023}
            },
            {
                "text": "Patient Jane Smith diagnosed with cardiovascular disease.",
                "metadata": {"patient_id": "P002", "type": "medical", "year": 2023}
            },
            {
                "text": "Research paper on diabetes management published in journal.",
                "metadata": {"patient_id": None, "type": "research", "year": 2022}
            }
        ]
        
        for doc in test_docs:
            chunks = chunker.chunk_text(doc["text"])
            embeddings = embedder.encode_batch(chunks)
            repo.add_chunks(chunks, embeddings, [doc["metadata"]] * len(chunks))
        
        yield search
        
        # Cleanup
        repo.delete_collection()
    
    def test_filtered_search_by_patient_id(self, search_service):
        """Test filtering by specific patient ID."""
        results = search_service.metadata_filtered_search(
            query="treatment",
            metadata_filters={"patient_id": "P001"},
            top_k=5
        )
        
        assert len(results) > 0
        # All results should be from P001
        for result in results:
            assert result.metadata.get("patient_id") == "P001"
    
    def test_filtered_search_by_document_type(self, search_service):
        """Test filtering by document type."""
        results = search_service.metadata_filtered_search(
            query="diabetes",
            metadata_filters={"type": "research"},
            top_k=5
        )
        
        # Should find research paper, not patient records
        assert len(results) > 0
        for result in results:
            assert result.metadata.get("type") == "research"
    
    def test_filtered_search_multiple_filters(self, search_service):
        """Test filtering with multiple criteria."""
        results = search_service.metadata_filtered_search(
            query="disease",
            metadata_filters={"type": "medical", "year": 2023},
            top_k=5
        )
        
        assert len(results) > 0
        for result in results:
            assert result.metadata.get("type") == "medical"
            assert result.metadata.get("year") == 2023
    
    def test_filtered_search_empty_query_raises_error(self, search_service):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.metadata_filtered_search(
                query="",
                metadata_filters={"patient_id": "P001"},
                top_k=5
            )
    
    def test_filtered_search_empty_filters_raises_error(self, search_service):
        """Test that empty filters raises ValueError."""
        with pytest.raises(ValueError, match="Metadata filters cannot be empty"):
            search_service.metadata_filtered_search(
                query="test",
                metadata_filters={},
                top_k=5
            )


class TestHybridSearch:
    """Test hybrid search combining vector, metadata, and keywords."""
    
    @pytest.fixture
    def search_service(self):
        """Create SearchService with rich test data."""
        chunker = ChunkingService(chunk_size=150, chunk_overlap=30)
        embedder = EmbeddingService()
        repo = ChromaRepository(collection_name="test_hybrid_search")
        search = SearchService(embedding_service=embedder, chroma_repository=repo)
        
        # Add test documents with varying characteristics
        test_docs = [
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
        
        for doc in test_docs:
            chunks = chunker.chunk_text(doc["text"])
            embeddings = embedder.encode_batch(chunks)
            repo.add_chunks(chunks, embeddings, [doc["metadata"]] * len(chunks))
        
        yield search
        
        # Cleanup
        repo.delete_collection()
    
    def test_hybrid_search_combines_all_signals(self, search_service):
        """Test that hybrid search uses vector + metadata + keywords."""
        results = search_service.hybrid_search(
            query="diabetes treatment",
            metadata_filters={"urgency": "high"},
            keywords=["urgent", "ICU"],
            top_k=3
        )
        
        assert len(results) > 0
        # First result should match all criteria (urgent diabetes case in ICU)
        top_result = results[0]
        assert "diabetes" in top_result.text.lower() or "insulin" in top_result.text.lower()
        assert top_result.metadata.get("urgency") == "high"
    
    def test_hybrid_search_keyword_boosting(self, search_service):
        """Test that keywords boost relevance scores."""
        # Search with keywords
        results_with_keywords = search_service.hybrid_search(
            query="insulin",
            keywords=["urgent", "ICU"],
            top_k=3
        )
        
        # Search without keywords
        results_without_keywords = search_service.semantic_search(
            query="insulin",
            top_k=3
        )
        
        # Rankings may differ due to keyword boosting
        assert len(results_with_keywords) > 0
        assert len(results_without_keywords) > 0
    
    def test_hybrid_search_custom_weights(self, search_service):
        """Test hybrid search with custom weight distribution."""
        results = search_service.hybrid_search(
            query="emergency treatment",
            metadata_filters={"urgency": "high"},
            keywords=["critical"],
            top_k=2,
            vector_weight=0.6,
            metadata_weight=0.3,
            keyword_weight=0.1
        )
        
        assert len(results) > 0
        for result in results:
            assert 0.0 <= result.score <= 1.0
    
    def test_hybrid_search_weights_must_sum_to_one(self, search_service):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            search_service.hybrid_search(
                query="test",
                top_k=3,
                vector_weight=0.5,
                metadata_weight=0.5,
                keyword_weight=0.5  # Total = 1.5, invalid!
            )
    
    def test_hybrid_search_no_filters_or_keywords(self, search_service):
        """Test hybrid search falls back to semantic when no extras."""
        results = search_service.hybrid_search(
            query="diabetes",
            metadata_filters=None,
            keywords=None,
            top_k=2
        )
        
        # Should still work, essentially semantic search
        assert len(results) > 0
        for result in results:
            assert isinstance(result, SearchResult)


class TestHelperFunctions:
    """Test helper functions for scoring and formatting."""
    
    @pytest.fixture
    def search_service(self):
        """Create minimal SearchService for testing helpers."""
        return SearchService()
    
    def test_calculate_metadata_score_perfect_match(self, search_service):
        """Test metadata scoring with perfect match."""
        result_meta = {"type": "medical", "year": 2023, "author": "Smith"}
        filter_meta = {"type": "medical", "year": 2023}
        
        score = search_service._calculate_metadata_score(result_meta, filter_meta)
        
        assert score == 1.0  # 2/2 matches
    
    def test_calculate_metadata_score_partial_match(self, search_service):
        """Test metadata scoring with partial match."""
        result_meta = {"type": "medical", "year": 2023, "author": "Smith"}
        filter_meta = {"type": "medical", "year": 2022}  # Year doesn't match
        
        score = search_service._calculate_metadata_score(result_meta, filter_meta)
        
        assert score == 0.5  # 1/2 matches
    
    def test_calculate_metadata_score_no_match(self, search_service):
        """Test metadata scoring with no matches."""
        result_meta = {"type": "tech", "year": 2020}
        filter_meta = {"type": "medical", "year": 2023}
        
        score = search_service._calculate_metadata_score(result_meta, filter_meta)
        
        assert score == 0.0  # 0/2 matches
    
    def test_calculate_metadata_score_empty_filters(self, search_service):
        """Test metadata scoring with no filters returns perfect score."""
        result_meta = {"type": "medical"}
        filter_meta = {}
        
        score = search_service._calculate_metadata_score(result_meta, filter_meta)
        
        assert score == 1.0
    
    def test_calculate_keyword_score_all_found(self, search_service):
        """Test keyword scoring when all keywords present."""
        text = "Patient requires urgent insulin therapy in ICU"
        keywords = ["urgent", "insulin", "ICU"]
        
        score = search_service._calculate_keyword_score(text, keywords)
        
        assert score == 1.0  # 3/3 found
    
    def test_calculate_keyword_score_partial_found(self, search_service):
        """Test keyword scoring with some keywords found."""
        text = "Patient requires urgent insulin therapy"
        keywords = ["urgent", "insulin", "surgery"]  # Surgery not present
        
        score = search_service._calculate_keyword_score(text, keywords)
        
        assert score == pytest.approx(0.667, abs=0.01)  # 2/3 found
    
    def test_calculate_keyword_score_none_found(self, search_service):
        """Test keyword scoring when no keywords present."""
        text = "Patient requires treatment"
        keywords = ["surgery", "emergency", "critical"]
        
        score = search_service._calculate_keyword_score(text, keywords)
        
        assert score == 0.0  # 0/3 found
    
    def test_calculate_keyword_score_case_insensitive(self, search_service):
        """Test that keyword matching is case-insensitive."""
        text = "URGENT patient needs INSULIN"
        keywords = ["urgent", "insulin"]
        
        score = search_service._calculate_keyword_score(text, keywords)
        
        assert score == 1.0  # Both found despite case difference
    
    def test_calculate_keyword_score_empty_keywords(self, search_service):
        """Test keyword scoring with no keywords returns perfect score."""
        text = "Some text"
        keywords = []
        
        score = search_service._calculate_keyword_score(text, keywords)
        
        assert score == 1.0
    
    def test_format_results_converts_distances_to_scores(self, search_service):
        """Test that distance values are converted to similarity scores."""
        # Simulate ChromaDB response format
        raw_results = {
            "ids": [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.2, 0.6]]  # Distance: lower = more similar
        }
        
        results = search_service._format_results(raw_results)
        
        assert len(results) == 2
        # Distance 0.2 → Similarity 1 - (0.2/2) = 0.9
        assert results[0].score == 0.9
        # Distance 0.6 → Similarity 1 - (0.6/2) = 0.7
        assert results[1].score == 0.7
    
    def test_format_results_handles_empty_response(self, search_service):
        """Test formatting empty results."""
        raw_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        results = search_service._format_results(raw_results)
        
        assert results == []


class TestSearchServiceStats:
    """Test stats retrieval."""
    
    def test_get_stats_returns_service_info(self):
        """Test that get_stats returns service configuration."""
        search = SearchService()
        stats = search.get_stats()
        
        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
        assert "vector_database" in stats
        assert "collection_stats" in stats
        
        assert stats["vector_database"] == "ChromaDB"
        assert stats["embedding_dimension"] == 384


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
