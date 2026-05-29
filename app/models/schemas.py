"""
Pydantic models for request/response validation

Why Pydantic?
- Automatic validation
- Clear API documentation (FastAPI uses these for OpenAPI)
- Type safety
- Easy serialization/deserialization
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, validator
from typing import List, Dict, Optional, Any
from datetime import datetime


class MetadataBase(BaseModel):
    """
    Base metadata model - required fields for all documents
    
    Why these fields?
    - source: Track where document came from (ocr, pdf, image)
    - page_number: Locate original content
    - chunk_index: Order chunks correctly
    - created_at: Temporal filtering
    - tags: Categorical filtering
    """
    source: str = Field(..., description="Source type: ocr, pdf, or image")
    page_number: int = Field(..., ge=1, description="Page number in original document")
    chunk_index: int = Field(default=0, ge=0, description="Index of chunk within document")
    created_at: str = Field(..., description="ISO timestamp of document creation")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    
    @validator('source')
    def validate_source(cls, v):
        """Why validate? Prevent invalid source types"""
        allowed_sources = ['ocr', 'pdf', 'image']
        if v not in allowed_sources:
            raise ValueError(f"Source must be one of {allowed_sources}")
        return v
    
    @validator('created_at')
    def validate_timestamp(cls, v):
        """Why validate? Ensure proper ISO format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("created_at must be valid ISO timestamp")
        return v


class IndexRequest(BaseModel):
    """Request model for indexing a document"""
    document_id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User who owns the document")
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    
    @validator('content')
    def validate_content(cls, v):
        """Why validate? Prevent empty documents"""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_12345",
                "user_id": "user_789",
                "content": "This is a medical invoice for patient John Doe...",
                "metadata": {
                    "source": "ocr",
                    "page_number": 1,
                    "chunk_index": 0,
                    "created_at": "2026-01-14T10:00:00Z",
                    "tags": ["medical", "invoice"]
                }
            }
        }


class IndexResponse(BaseModel):
    """Response model for successful indexing"""
    message: str
    document_id: str
    chunks_created: int
    chunk_ids: List[str]


class SearchType(str, Enum):
    """Search mode for unified POST /vector/search."""

    semantic = "semantic"
    filtered = "filtered"
    hybrid = "hybrid"


class VectorSearchRequest(BaseModel):
    """
    Unified search body (assignment §7: POST /vector/search).

    Use search_type to select semantic, metadata-filtered, or hybrid search.
    """

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    search_type: SearchType = Field(
        default=SearchType.semantic,
        description="semantic | filtered | hybrid",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (required when search_type=filtered)",
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for hybrid search",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords that must appear in chunk text (hybrid)",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    
    @validator('query')
    def validate_query(cls, v):
        """Why validate? Prevent empty queries"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "diabetes treatment and insulin therapy",
                "top_k": 10
            }
        }


class MetadataSearchRequest(BaseModel):
    """Request model for metadata-filtered search"""
    query: str = Field(..., min_length=1, description="Search query")
    filters: Dict[str, Any] = Field(..., description="Metadata filters")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "patient medical records",
                "filters": {
                    "source": "ocr",
                    "tags": {"$in": ["medical", "radiology"]}
                },
                "top_k": 10
            }
        }


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search (vector + metadata + keywords)"""
    query: str = Field(..., min_length=1, description="Search query")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    keywords: Optional[List[str]] = Field(None, description="Keywords that must be present")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "diabetes treatment plans",
                "metadata_filters": {
                    "source": "ocr",
                    "tags": {"$in": ["medical"]}
                },
                "keywords": ["insulin", "therapy"],
                "top_k": 10
            }
        }


class SearchResult(BaseModel):
    """Individual search result"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    distance: float = Field(..., description="Distance from query (lower = more similar)")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (higher = more similar)")


class SearchResponse(BaseModel):
    """Response model for search operations"""
    results: List[SearchResult]
    count: int
    query: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk_id": "doc123_chunk_0",
                        "content": "Patient diagnosed with diabetes...",
                        "metadata": {
                            "document_id": "doc123",
                            "source": "ocr",
                            "tags": ["medical"]
                        },
                        "distance": 0.15,
                        "similarity_score": 0.85
                    }
                ],
                "count": 1,
                "query": "diabetes treatment"
            }
        }


class StatsResponse(BaseModel):
    """Response model for database statistics"""

    total_chunks: int
    collection_name: str
    embedding_dimension: Optional[int] = None
    persist_directory: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "total_chunks": 1250,
                "collection_name": "documents",
                "embedding_dimension": 384,
                "persist_directory": "./data/chroma_db",
                "chunk_size": 512,
                "chunk_overlap": 50,
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    status_code: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Query cannot be empty",
                "status_code": 400
            }
        }
