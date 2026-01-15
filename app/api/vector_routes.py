"""
Vector API Routes: REST endpoints for document indexing and search

Provides 5 HTTP endpoints:
1. POST /vector/index - Index documents into vector database
2. POST /vector/search - Semantic search (basic)
3. POST /vector/search/filtered - Search with metadata filtering
4. POST /vector/search/hybrid - Advanced hybrid search
5. GET /vector/stats - Database statistics

Why FastAPI?
- Automatic request/response validation with Pydantic
- OpenAPI/Swagger documentation generation
- Async support for high concurrency
- Type hints and editor support
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict
from app.models.schemas import (
    IndexRequest,
    SearchRequest,
    FilteredSearchRequest,
    HybridSearchRequest,
    SearchResponse,
    IndexResponse,
    StatsResponse
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.repositories.chroma_repository import ChromaRepository


# Create router
router = APIRouter()

# Initialize services (singleton pattern)
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
chroma_repository = ChromaRepository()
search_service = SearchService(
    embedding_service=embedding_service,
    chroma_repository=chroma_repository
)


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index a document",
    description="""
    Index a document into the vector database.
    
    Process:
    1. Split text into chunks with overlap
    2. Generate embeddings for each chunk
    3. Store chunks with vectors and metadata in ChromaDB
    
    Example:
    ```json
    {
      "text": "Patient John Doe has Type 2 diabetes...",
      "metadata": {
        "patient_id": "P001",
        "document_type": "medical_report",
        "date": "2024-01-15"
      }
    }
    ```
    """
)
async def index_document(request: IndexRequest) -> IndexResponse:
    """
    Index a document into the vector database.
    
    Args:
        request: IndexRequest with text and metadata
        
    Returns:
        IndexResponse with success status and chunk count
        
    Raises:
        HTTPException: 400 if text is empty, 500 if indexing fails
    """
    try:
        # Validate text
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        # Step 1: Chunk the text
        chunks = chunking_service.chunk_text(request.text)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to generate chunks from text"
            )
        
        # Step 2: Generate embeddings for chunks
        embeddings = embedding_service.encode_batch(chunks)
        
        # Step 3: Prepare metadata for each chunk
        metadata_list = [request.metadata or {}] * len(chunks)
        
        # Step 4: Generate document ID
        import uuid
        document_id = str(uuid.uuid4())
        
        # Step 5: Store in ChromaDB
        chunk_ids = chroma_repository.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadata_list,
            document_id=document_id
        )
        
        # Return success response
        return IndexResponse(
            status="success",
            chunks_indexed=len(chunks),
            message=f"Document indexed successfully with {len(chunks)} chunks (ID: {document_id})"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description="""
    Perform semantic (vector similarity) search across all documents.
    
    Uses pure vector similarity - no filtering or keyword matching.
    Best for exploratory queries where you want to find semantically similar content.
    
    Example:
    ```json
    {
      "query": "diabetes treatment",
      "top_k": 5
    }
    ```
    
    Returns documents ranked by semantic similarity score.
    """
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform semantic search.
    
    Args:
        request: SearchRequest with query and top_k
        
    Returns:
        SearchResponse with ranked results
        
    Raises:
        HTTPException: 400 if query invalid, 404 if no results, 500 if search fails
    """
    try:
        # Perform search
        results = search_service.semantic_search(
            query=request.query,
            top_k=request.top_k
        )
        
        # Check if results found
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No results found for query: {request.query}"
            )
        
        # Return results
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_type="semantic"
        )
        
    except ValueError as e:
        # Validation errors from service layer
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/search/filtered",
    response_model=SearchResponse,
    summary="Filtered search",
    description="""
    Search with metadata filtering - pre-filter by metadata then semantic search.
    
    Best for:
    - Searching within specific documents (patient records, contracts, etc.)
    - Time-based filtering (recent documents only)
    - Access control (user-specific documents)
    
    Example:
    ```json
    {
      "query": "treatment plan",
      "metadata_filters": {
        "patient_id": "P001",
        "document_type": "medical_report"
      },
      "top_k": 5
    }
    ```
    
    Filters applied BEFORE vector search for better performance.
    """
)
async def search_filtered(request: FilteredSearchRequest) -> SearchResponse:
    """
    Perform metadata-filtered search.
    
    Args:
        request: FilteredSearchRequest with query, filters, and top_k
        
    Returns:
        SearchResponse with filtered and ranked results
        
    Raises:
        HTTPException: 400 if query/filters invalid, 404 if no results, 500 if search fails
    """
    try:
        # Perform filtered search
        results = search_service.metadata_filtered_search(
            query=request.query,
            metadata_filters=request.metadata_filters,
            top_k=request.top_k
        )
        
        # Check if results found
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No results found matching query and filters"
            )
        
        # Return results
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_type="filtered",
            filters_applied=request.metadata_filters
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Filtered search failed: {str(e)}"
        )


@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    summary="Hybrid search",
    description="""
    Advanced search combining vector similarity, metadata matching, and keyword presence.
    
    Scoring formula:
    final_score = (vector_weight × semantic_score) + 
                  (metadata_weight × metadata_match_score) + 
                  (keyword_weight × keyword_presence_score)
    
    Best for complex queries with multiple requirements.
    
    Example:
    ```json
    {
      "query": "insulin dosage",
      "metadata_filters": {
        "urgency": "high"
      },
      "keywords": ["emergency", "ICU"],
      "top_k": 5,
      "weights": {
        "vector": 0.5,
        "metadata": 0.3,
        "keyword": 0.2
      }
    }
    ```
    
    Results ranked by combined score for maximum relevance.
    """
)
async def search_hybrid(request: HybridSearchRequest) -> SearchResponse:
    """
    Perform hybrid search with combined scoring.
    
    Args:
        request: HybridSearchRequest with query, filters, keywords, weights
        
    Returns:
        SearchResponse with hybrid-ranked results
        
    Raises:
        HTTPException: 400 if parameters invalid, 404 if no results, 500 if search fails
    """
    try:
        # Extract weights (use defaults if not provided)
        weights = request.weights or {}
        vector_weight = weights.get("vector", 0.5)
        metadata_weight = weights.get("metadata", 0.3)
        keyword_weight = weights.get("keyword", 0.2)
        
        # Perform hybrid search
        results = search_service.hybrid_search(
            query=request.query,
            metadata_filters=request.metadata_filters,
            keywords=request.keywords,
            top_k=request.top_k,
            vector_weight=vector_weight,
            metadata_weight=metadata_weight,
            keyword_weight=keyword_weight
        )
        
        # Check if results found
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No results found for hybrid search"
            )
        
        # Return results
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            search_type="hybrid",
            filters_applied=request.metadata_filters,
            keywords_used=request.keywords
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get database statistics",
    description="""
    Retrieve statistics about the vector database and search service.
    
    Returns:
    - Embedding model information
    - Vector dimension
    - Database type
    - Collection statistics (document count, etc.)
    
    Useful for monitoring, health checks, and debugging.
    """
)
async def get_stats() -> StatsResponse:
    """
    Get database and service statistics.
    
    Returns:
        StatsResponse with configuration and stats
        
    Raises:
        HTTPException: 500 if stats retrieval fails
    """
    try:
        # Get stats from search service
        stats = search_service.get_stats()
        
        return StatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}"
        )
