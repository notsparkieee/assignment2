"""
EmbeddingService: Text to vector conversion using sentence-transformers

Purpose: Convert text into semantic vector representations for similarity search.

What are embeddings?
- Mathematical representation of text meaning
- Similar texts → similar vectors
- Enables semantic search (meaning-based, not keyword-based)

Example:
    embedder = EmbeddingService()
    embedding = embedder.encode("diabetes treatment")
    # Returns: [0.23, -0.45, 0.67, ..., 0.89] (384 floats)
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import settings


class EmbeddingService:
    """
    Converts text to vector embeddings using sentence-transformers.
    
    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Speed: ~1000 sentences/sec on CPU
    - Quality: 95% of BERT at 2x speed
    - Size: 80MB
    
    Why sentence-transformers?
    - Pre-trained on semantic similarity tasks
    - Fast inference (CPU-friendly)
    - Good quality embeddings
    - Free and open-source
    
    Alternative models:
    - all-mpnet-base-v2: Better accuracy (768-dim) but slower
    - paraphrase-MiniLM-L6-v2: Good for paraphrase detection
    - all-distilroberta-v1: Good accuracy, 768-dim
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize embedding model (loads once, expensive operation).
        
        Why singleton pattern?
        - Model loading takes ~2 seconds
        - Reuse same model for all embeddings
        - Saves memory (model is ~80MB)
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded successfully!")
        print(f"   Embedding dimension: {self.dimension}")
        print(f"   Max sequence length: {self.model.max_seq_length}")
    
    def encode(self, text: str) -> List[float]:
        """
        Convert single text to embedding vector.
        
        Process:
        1. Tokenize text (words → token IDs)
        2. Pass through neural network
        3. Get vector representation (mean pooling of last layer)
        
        Time Complexity: O(n) where n = text length
        
        Args:
            text: Input text to encode
            
        Returns:
            List of floats (384 dimensions for MiniLM)
            
        Example:
            encode("diabetes treatment")
            → [0.234, -0.456, 0.123, ..., 0.789]
            
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot encode empty text")
        
        # Why convert_to_numpy=True?
        # - Faster than Python lists for numerical operations
        # - Compatible with Chroma/FAISS
        # - Efficient memory usage
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        # Convert to list for JSON serialization
        return embedding.tolist()
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Encode multiple texts efficiently using batch processing.
        
        Why batch encoding?
        - 10x faster than encoding one-by-one
        - GPU/CPU can process multiple sentences simultaneously
        - Reduces overhead of model inference
        
        Performance comparison:
        - Sequential: 10ms × 1000 texts = 10,000ms (10 seconds)
        - Batch: 1000ms total = 1 second (10x faster!)
        
        Args:
            texts: List of texts to encode
            batch_size: Number of texts per batch (default: 32)
            show_progress: Show progress bar (default: False)
            
        Returns:
            List of embedding vectors
            
        Why batch_size=32?
        - Balance between speed and memory
        - Larger batches: Faster but more memory
        - Smaller batches: Slower but less memory
        - 32 is optimal for most CPUs
        
        Example:
            encode_batch(["text1", "text2", "text3"])
            → [[emb1], [emb2], [emb3]]
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts to encode")
        
        # Batch encode with normalization
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine Similarity Formula:
        similarity = (A · B) / (||A|| × ||B||)
        
        Where:
        - A · B = dot product (sum of element-wise multiplication)
        - ||A|| = L2 norm (magnitude of vector A)
        - ||B|| = L2 norm (magnitude of vector B)
        
        Returns:
        - 1.0  = identical meaning
        - 0.5  = somewhat similar
        - 0.0  = unrelated
        - -1.0 = opposite meaning (rare for text)
        
        Why cosine over Euclidean?
        - Measures angle (direction), not distance (magnitude)
        - Normalized: always between -1 and 1
        - Standard for text embeddings
        - "cat" vs "cats" → high similarity (same direction)
        
        Example:
            emb1 = encode("diabetes")
            emb2 = encode("blood sugar")
            similarity(emb1, emb2) → 0.78 (similar!)
            
            emb3 = encode("pizza")
            similarity(emb1, emb3) → 0.12 (different!)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Dot product: sum of element-wise multiplication
        dot_product = np.dot(emb1, emb2)
        
        # L2 norms (magnitude of vectors)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def get_dimension(self) -> int:
        """
        Get embedding vector dimension.
        
        Why needed?
        - Chroma needs dimension for index creation
        - Validation: ensure all embeddings same size
        - Documentation: what dimension was used?
        
        Returns:
            Embedding dimension (384 for MiniLM)
        """
        return self.dimension
    
    def get_model_info(self) -> dict:
        """
        Get model metadata and configuration.
        
        Why useful?
        - Documentation: which model version used?
        - Debugging: embedding dimension mismatch?
        - Reproducibility: use same model for consistency
        
        Returns:
            Dict with model name, dimension, and max sequence length
        """
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "pooling_mode": "mean"  # How tokens are aggregated
        }
