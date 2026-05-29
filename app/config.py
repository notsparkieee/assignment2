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
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Why Pydantic BaseSettings?
    - Automatic type validation (catches errors early)
    - Loads from .env files automatically
    - Default values if env vars not set
    - Easy to test (can override settings in tests)
    
    Usage:
    1. Copy .env.example to .env
    2. Customize values in .env
    3. Settings auto-loaded on import
    """
    
    # ==========================================
    # Application Settings
    # ==========================================
    APP_NAME: str = "Vector DB & Metadata Search System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # ==========================================
    # Chunking Configuration
    # ==========================================
    CHUNK_SIZE: int = 512
    """
    Number of characters per chunk.
    
    Why 512?
    - Balance: Enough context vs precise matching
    - Most embedding models handle 512 tokens well
    - Larger (1024): More context, less precision
    - Smaller (256): More precision, less context
    
    Trade-offs:
    - Storage: Smaller chunks = more chunks = more storage
    - Search: Smaller chunks = more precise but may miss context
    - Speed: More chunks = longer indexing time
    """
    
    CHUNK_OVERLAP: int = 50
    """
    Number of overlapping characters between chunks.
    
    Why 50 (10% of 512)?
    - Prevents sentence/context split at boundaries
    - Example: "...diabetes." | "Treatment..." 
              → "...diabetes. Treat" | "Treatment..."
    - Ensures semantic continuity
    
    Trade-offs:
    - More overlap: Better context, more storage
    - Less overlap: Less storage, risk losing context
    """
    
    # ==========================================
    # Embedding Configuration
    # ==========================================
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    """
    Sentence-transformer model for generating embeddings.
    
    Why all-MiniLM-L6-v2?
    - Fast: 384 dimensions (vs 768 for BERT)
    - Accurate: 95% of BERT quality at 2x speed
    - Small: 80MB model size
    - Free: No API costs
    - CPU-friendly: ~1000 sentences/sec on CPU
    
    Alternatives:
    - all-mpnet-base-v2: Better accuracy (768-dim) but slower
    - paraphrase-MiniLM-L6-v2: Good for paraphrase tasks
    - all-distilroberta-v1: Good accuracy, 768-dim
    
    Performance Comparison (CPU):
    - MiniLM-L6: ~1000 sent/sec, 384-dim
    - MPNet: ~400 sent/sec, 768-dim
    - BERT-base: ~200 sent/sec, 768-dim
    """
    
    EMBEDDING_DIMENSION: int = 384
    """
    Dimension of embedding vectors (must match model output).
    
    Why 384?
    - Matches all-MiniLM-L6-v2 output
    - 2x faster than 768-dim models
    - Still captures semantic meaning well
    
    Note: Change this if you change EMBEDDING_MODEL
    """
    
    # ==========================================
    # Chroma Database Configuration
    # ==========================================
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    """
    Directory for persistent Chroma storage.
    
    Why persist?
    - Data survives restarts (production requirement)
    - No re-indexing needed after restart
    - Fast startup (load existing index)
    
    Production:
    - Use absolute path or mounted volume
    - Ensure directory has write permissions
    - Backup regularly
    """
    
    CHROMA_COLLECTION_NAME: str = "documents"
    """
    Name of the Chroma collection.
    
    Why needed?
    - Can have multiple collections (e.g., 'documents', 'images')
    - Logical separation of data
    - Easy to reset/rebuild specific collections
    """
    
    CHROMA_DISTANCE_FUNCTION: str = "cosine"
    """
    Distance function for similarity search.
    
    Why cosine?
    - Standard for text embeddings
    - Measures angle between vectors (direction)
    - Normalized: always between 0 and 1
    - Ignores magnitude (only cares about direction)
    
    Alternatives:
    - l2 (Euclidean): Measures absolute distance
    - ip (Inner Product): Dot product, not normalized
    
    Cosine is best for text because:
    - "cat" and "cats" have similar direction
    - Length normalization handled automatically
    """
    
    # ==========================================
    # Search Configuration
    # ==========================================
    DEFAULT_TOP_K: int = 10
    """
    Default number of search results to return.
    
    Why 10?
    - Standard in Information Retrieval
    - Good balance: quality vs quantity
    - User attention span (top 10 most relevant)
    - Can be overridden per request
    """
    
    MAX_TOP_K: int = 100
    """
    Maximum allowed results per query.
    
    Why limit to 100?
    - Prevents abuse/expensive queries
    - Diminishing returns after ~50 results
    - API performance (large responses slow)
    - Most users only check top 10-20
    """
    
    # ==========================================
    # API Configuration
    # ==========================================
    API_V1_PREFIX: str = "/api/v1"
    """
    API version prefix for routes.
    
    Why versioning?
    - Backward compatibility (v1, v2 can coexist)
    - Gradual migration for clients
    - Clear API evolution
    """
    
    CORS_ORIGINS: List[str] = ["*"]
    """
    Allowed CORS origins (comma-separated in .env).
    
    Development: ["*"] allows all origins
    Production: ["https://yourapp.com", "https://api.yourapp.com"]
    
    Why restrict?
    - Security: Prevent unauthorized frontend access
    - Control: Know who's using your API
    """
    
    # ==========================================
    # Pydantic Configuration
    # ==========================================
    class Config:
        env_file = ".env"
        case_sensitive = True
        """
        Why case_sensitive?
        - Prevents confusion (CHUNK_SIZE vs chunk_size)
        - Explicit about what you're setting
        - Matches environment variable conventions
        """


# ==========================================
# Global Settings Instance
# ==========================================
settings = Settings()
"""
Why global instance?
- Single source of truth across application
- Settings loaded once at startup
- Easy to import: from app.config import settings

Usage:
    from app.config import settings
    
    chunk_size = settings.CHUNK_SIZE
    model_name = settings.EMBEDDING_MODEL
"""
