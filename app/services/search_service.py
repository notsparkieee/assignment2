"""
SearchService: Orchestrates semantic, filtered, and hybrid search strategies

Purpose: Provide multiple search modes for different use cases.

Why three strategies?
- Semantic: Best for exploratory "find similar" queries
- Filtered: Best for "find in specific subset" queries  
- Hybrid: Best for complex queries with multiple constraints

Example:
    search = SearchService()
    results = search.semantic_search("diabetes treatment", top_k=5)
"""

from typing import List, Dict, Optional
from app.services.embedding_service import EmbeddingService
from app.repositories.chroma_repository import ChromaRepository
from app.models.schemas import SearchResult


class SearchService:
    """
    High-level search orchestration combining embedding and vector DB.
    
    Architecture:
    - Uses EmbeddingService to convert queries to vectors
    - Uses ChromaRepository to search vector database
    - Formats raw results into clean API responses
    
    Why this design?
    - Service layer abstracts search complexity
    - Can swap embedding models without changing search logic
    - Can swap vector DBs without changing API
    - Testable without HTTP layer
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_repository: Optional[ChromaRepository] = None
    ):
        """
        Initialize search service with dependencies.
        
        Args:
            embedding_service: Optional custom embedding service (default: create new)
            chroma_repository: Optional custom repository (default: create new)
            
        Why optional dependencies?
        - Default: Convenient for production use
        - Custom: Allows dependency injection for testing
        
        Example:
            # Production use
            search = SearchService()
            
            # Testing with mocks
            mock_embedder = MockEmbeddingService()
            search = SearchService(embedding_service=mock_embedder)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.chroma_repository = chroma_repository or ChromaRepository()
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Pure vector similarity search - finds semantically similar content.
        
        Algorithm:
        1. Convert query text → query vector (384 dimensions)
        2. Search ChromaDB for nearest neighbor vectors
        3. Format and return results
        
        Time Complexity: O(log n) with HNSW index
        Space Complexity: O(k) for storing top_k results
        
        Use cases:
        - "Find documents about heart disease" (broad, exploratory)
        - "Show similar research papers" (content recommendation)
        - "What else talks about this topic?" (discovery)
        
        Example:
            Query: "diabetes treatment"
            Finds: "insulin therapy", "blood glucose management", "Type 2 care"
            Why: Semantic similarity, not just keyword matching
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: 5)
            
        Returns:
            List of SearchResult objects with text, score, metadata
            
        Raises:
            ValueError: If query is empty or top_k < 1
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        # Step 1: Convert query to vector
        query_vector = self.embedding_service.encode(query)
        
        # Step 2: Search vector database
        raw_results = self.chroma_repository.semantic_search(
            query_embedding=query_vector,
            n_results=top_k
        )
        
        # Step 3: Format results
        return self._format_results(raw_results)
    
    def metadata_filtered_search(
        self,
        query: str,
        metadata_filters: Dict,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Pre-filter by metadata, then semantic search on filtered subset.
        
        Algorithm:
        1. Filter database by metadata constraints (reduces n to k documents)
        2. Convert query → vector
        3. Search only filtered subset
        4. Format and return results
        
        Time Complexity: O(k·log k) where k = filtered size, k << n
        Space Complexity: O(k) for filtered results
        
        Why filter first?
        - Performance: Search 100 docs instead of 10,000
        - Relevance: Only search relevant subset
        - Privacy: Enforce access control (user_id, department, etc.)
        
        Use cases:
        - "Find diabetes info in patient P001's records only"
        - "Search contracts signed in 2023"
        - "Show research papers from Stanford only"
        
        Example:
            Query: "treatment plan"
            Filters: {"patient_id": "P001", "document_type": "medical_report"}
            Result: Only searches P001's medical reports, not all documents
        
        Args:
            query: Search query text
            metadata_filters: Dict of metadata constraints (e.g., {"year": 2023})
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects matching both query AND filters
            
        Raises:
            ValueError: If query empty, filters empty, or top_k < 1
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not metadata_filters:
            raise ValueError("Metadata filters cannot be empty (use semantic_search instead)")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        # Step 1: Convert query to vector
        query_vector = self.embedding_service.encode(query)
        
        # Step 2: Search with metadata filtering
        # ChromaDB handles filter-then-search internally
        raw_results = self.chroma_repository.metadata_filtered_search(
            query_embedding=query_vector,
            filters=metadata_filters,
            n_results=top_k
        )
        
        # Step 3: Format results
        return self._format_results(raw_results)
    
    def hybrid_search(
        self,
        query: str,
        metadata_filters: Optional[Dict] = None,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        vector_weight: float = 0.5,
        metadata_weight: float = 0.3,
        keyword_weight: float = 0.2
    ) -> List[SearchResult]:
        """
        Combined scoring: vector similarity + metadata matching + keyword presence.
        
        Algorithm:
        1. Get semantic search results (vector scores)
        2. Calculate metadata match scores for each result
        3. Calculate keyword presence scores for each result
        4. Combine: final_score = w1×vector + w2×metadata + w3×keywords
        5. Re-rank by combined score
        
        Time Complexity: O(k·m) where k=top_k, m=avg metadata fields
        Space Complexity: O(k) for results
        
        Scoring weights (default):
        - Vector: 50% (semantic meaning most important)
        - Metadata: 30% (structure and filtering)
        - Keywords: 20% (exact term matching)
        
        Why hybrid?
        - Semantic search alone misses exact term importance
        - Metadata alone misses semantic meaning
        - Keywords alone miss synonyms and context
        - Combined = best of all worlds
        
        Use cases:
        - "Find urgent cardiac cases from this month with 'surgery' mentioned"
        - "Show high-priority contracts containing 'termination clause'"
        - "Research papers about 'CRISPR' from top universities in 2023"
        
        Example:
            Query: "insulin dosage"
            Metadata filters: {"urgency": "high"}
            Keywords: ["emergency", "ICU"]
            
            Result scoring:
            - Chunk A: "Emergency insulin protocol in ICU..."
              Vector: 0.85, Metadata: 1.0 (has urgency=high), Keywords: 1.0 (both found)
              Final: 0.5×0.85 + 0.3×1.0 + 0.2×1.0 = 0.925 ← TOP RESULT
            
            - Chunk B: "Standard insulin dosage guidelines..."
              Vector: 0.90, Metadata: 0.0 (no urgency), Keywords: 0.0 (none found)
              Final: 0.5×0.90 + 0.3×0.0 + 0.2×0.0 = 0.45 ← LOWER RANK
        
        Args:
            query: Search query text
            metadata_filters: Optional dict for pre-filtering
            keywords: Optional list of exact terms to boost
            top_k: Number of results to return
            vector_weight: Weight for semantic similarity (default: 0.5)
            metadata_weight: Weight for metadata matching (default: 0.3)
            keyword_weight: Weight for keyword presence (default: 0.2)
            
        Returns:
            List of SearchResult objects ranked by combined score
            
        Raises:
            ValueError: If query empty, top_k < 1, or weights don't sum to 1.0
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        # Validate weights sum to 1.0 (with small tolerance for floating point)
        total_weight = vector_weight + metadata_weight + keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.2f} "
                f"({vector_weight} + {metadata_weight} + {keyword_weight})"
            )
        
        # Step 1: Get base results from semantic or filtered search
        if metadata_filters:
            # If filters provided, use filtered search
            base_results = self.metadata_filtered_search(
                query=query,
                metadata_filters=metadata_filters,
                top_k=top_k * 2  # Get more results to re-rank
            )
        else:
            # No filters, use pure semantic search
            base_results = self.semantic_search(
                query=query,
                top_k=top_k * 2  # Get more results to re-rank
            )
        
        # If no results, return empty list
        if not base_results:
            return []
        
        # Step 2: Calculate hybrid scores for each result
        hybrid_results = []
        
        for result in base_results:
            # A) Vector similarity score (already normalized 0-1)
            vector_score = result.score
            
            # B) Metadata match score
            metadata_score = self._calculate_metadata_score(
                result_metadata=result.metadata,
                filter_metadata=metadata_filters or {}
            )
            
            # C) Keyword presence score
            keyword_score = self._calculate_keyword_score(
                text=result.text,
                keywords=keywords or []
            )
            
            # Combine scores with weights
            final_score = (
                vector_weight * vector_score +
                metadata_weight * metadata_score +
                keyword_weight * keyword_score
            )
            
            # Create new result with hybrid score
            hybrid_result = SearchResult(
                text=result.text,
                score=final_score,
                metadata=result.metadata
            )
            hybrid_results.append(hybrid_result)
        
        # Step 3: Re-rank by hybrid score (highest first)
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Step 4: Return top_k results
        return hybrid_results[:top_k]
    
    def _format_results(self, raw_results: Dict) -> List[SearchResult]:
        """
        Convert ChromaDB's nested result format to clean SearchResult objects.
        
        ChromaDB returns:
        {
            "ids": [["id1", "id2", ...]],
            "documents": [["text1", "text2", ...]],
            "metadatas": [[{meta1}, {meta2}, ...]],
            "distances": [[0.15, 0.23, ...]]
        }
        
        We convert to:
        [
            SearchResult(text="text1", score=0.85, metadata={meta1}),
            SearchResult(text="text2", score=0.77, metadata={meta2}),
            ...
        ]
        
        Why format?
        - API clients need clean, flat structure
        - Distance → Similarity conversion (1 - distance)
        - Remove unnecessary nesting
        - Type-safe with Pydantic models
        
        Args:
            raw_results: Raw response from ChromaDB
            
        Returns:
            List of SearchResult objects
        """
        # ChromaDB wraps everything in lists (batch support)
        # We always query single batch, so unwrap first element
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        
        # Convert to SearchResult objects
        results = []
        for text, distance, metadata in zip(documents, distances, metadatas):
            # Convert distance to similarity score
            # Distance: 0=identical, 2=opposite (cosine distance)
            # Similarity: 1=identical, 0=opposite
            similarity_score = 1.0 - (distance / 2.0)
            
            result = SearchResult(
                text=text,
                score=round(similarity_score, 4),  # Round to 4 decimals
                metadata=metadata or {}
            )
            results.append(result)
        
        return results
    
    def _calculate_metadata_score(
        self,
        result_metadata: Dict,
        filter_metadata: Dict
    ) -> float:
        """
        Calculate how well result metadata matches filter criteria.
        
        Scoring:
        - Each matching field: +1 point
        - Total score: matches / total_filters
        - Range: 0.0 (no matches) to 1.0 (all match)
        
        Example:
            result_metadata = {"type": "research", "year": 2023, "author": "Smith"}
            filter_metadata = {"type": "research", "year": 2023}
            
            Matches: 2 out of 2 filters
            Score: 2/2 = 1.0
        
        Args:
            result_metadata: Metadata from search result
            filter_metadata: Metadata filters used in query
            
        Returns:
            Score from 0.0 to 1.0
        """
        # If no filters, return perfect score
        if not filter_metadata:
            return 1.0
        
        matches = 0
        total_filters = len(filter_metadata)
        
        for key, filter_value in filter_metadata.items():
            result_value = result_metadata.get(key)
            
            # Check if values match
            if result_value == filter_value:
                matches += 1
        
        return matches / total_filters if total_filters > 0 else 1.0
    
    def _calculate_keyword_score(
        self,
        text: str,
        keywords: List[str]
    ) -> float:
        """
        Calculate keyword presence score (exact term matching).
        
        Scoring:
        - Each keyword found: +1 point
        - Total score: found_keywords / total_keywords
        - Range: 0.0 (none found) to 1.0 (all found)
        - Case-insensitive matching
        
        Example:
            text = "Patient requires urgent insulin therapy in ICU"
            keywords = ["urgent", "ICU", "surgery"]
            
            Found: "urgent" ✓, "ICU" ✓, "surgery" ✗
            Score: 2/3 = 0.67
        
        Args:
            text: Text to search for keywords
            keywords: List of terms to find
            
        Returns:
            Score from 0.0 to 1.0
        """
        # If no keywords, return perfect score
        if not keywords:
            return 1.0
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        found = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found += 1
        
        return found / len(keywords)
    
    def get_stats(self) -> Dict:
        """
        Return statistics about the search service and underlying database.
        
        Useful for:
        - Health checks: Is database accessible?
        - Monitoring: How many documents indexed?
        - Debugging: What model is being used?
        
        Returns:
            Dict with service configuration and database stats
        """
        return {
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.dimension,
            "vector_database": "ChromaDB",
            "collection_stats": self.chroma_repository.get_stats()
        }
