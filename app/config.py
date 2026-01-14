"""
Configuration settings for the Vector Database System

This module contains all configuration parameters that can be customized
for different environments (development, production, testing).

Why centralized config?
- Single source of truth
- Easy to modify settings without touching code
- Environment-specific configurations
- Better security (secrets in env vars)
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Why Pydantic BaseSettings?
    - Automatic type validation
    - Loads from .env files
    - Default values if env vars not set
    """
    
    # Application Settings
    APP_NAME: str = "Vector Database & Metadata-Driven Search System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512  # Why 512? Balance between context and granularity
    CHUNK_OVERLAP: int = 50  # Why 50? Prevents sentence cut-off at boundaries
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Why MiniLM? Fast & accurate
    EMBEDDING_DIMENSION: int = 384  # MiniLM output dimension
    
    # Chroma Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_DISTANCE_FUNCTION: str = "cosine"  # Why cosine? Best for text embeddings
    
    # Search Configuration
    DEFAULT_TOP_K: int = 10  # Default number of results to return
    MAX_TOP_K: int = 100  # Maximum allowed results
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]  # In production, specify exact origins
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
