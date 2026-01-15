"""
Pydantic models for request/response validation

Why Pydantic?
- Automatic validation
- Clear API documentation (FastAPI uses these for OpenAPI)
- Type safety
- Easy serialization/deserialization
"""

from pydantic import BaseModel, Field, validator
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
    text: str = Field(..., min_length=1, description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    
    @validator('text')
    def validate_text(cls, v):
        """Why validate? Prevent empty documents"""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient John Doe has Type 2 diabetes...",
                "metadata": {
                    "patient_id": "P001",
                    "document_type": "medical_report",
                    "date": "2024-01-15"
                }
            }
        }


class IndexResponse(BaseModel):
    """Response model for successful indexing"""
    status: str
    chunks_indexed: int
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "chunks_indexed": 4,
                "message": "Document indexed successfully with 4 chunks"
            }
        }


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    
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
                "top_k": 5
            }
        }


class FilteredSearchRequest(BaseModel):
    """Request model for metadata-filtered search"""
    query: str = Field(..., min_length=1, description="Search query")
    metadata_filters: Dict[str, Any] = Field(..., description="Metadata filters")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "treatment plan",
                "metadata_filters": {
                    "patient_id": "P001",
                    "document_type": "medical_report"
                },
                "top_k": 5
            }
        }


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search (vector + metadata + keywords)"""
    query: str = Field(..., min_length=1, description="Search query")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    keywords: Optional[List[str]] = Field(None, description="Keywords to boost")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    weights: Optional[Dict[str, float]] = Field(None, description="Scoring weights")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class SearchResult(BaseModel):
    """Individual search result"""
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (higher = more similar)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient diagnosed with diabetes...",
                "score": 0.87,
                "metadata": {
                    "patient_id": "P001",
                    "document_type": "medical_report"
                }
            }
        }


class SearchResponse(BaseModel):
    """Response model for search operations"""
    results: List[SearchResult]
    query: str
    total_results: int
    search_type: str = Field(default="semantic", description="Type of search performed")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filters that were applied")
    keywords_used: Optional[List[str]] = Field(None, description="Keywords that were used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "Patient diagnosed with diabetes...",
                        "score": 0.87,
                        "metadata": {
                            "patient_id": "P001"
                        }
                    }
                ],
                "query": "diabetes treatment",
                "total_results": 1,
                "search_type": "semantic"
            }
        }


class StatsResponse(BaseModel):
    """Response model for database statistics"""
    embedding_model: str
    embedding_dimension: int
    vector_database: str
    collection_stats: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "vector_database": "ChromaDB",
                "collection_stats": {
                    "total_documents": 156,
                    "collection_name": "vector_store"
                }
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
