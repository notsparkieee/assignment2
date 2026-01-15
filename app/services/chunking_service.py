"""
ChunkingService: Text splitting with configurable overlap

Purpose: Split large documents into smaller chunks for embedding generation.

Why chunking?
- Embedding models have token limits (typically 512 tokens)
- Smaller chunks = more precise search results
- Overlap prevents context loss at boundaries

Example:
    chunker = ChunkingService(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text("Long document text...")
"""

from typing import List, Dict
from app.config import settings


class ChunkingService:
    """
    Splits text into overlapping chunks for embedding generation.
    
    Algorithm: Sliding window with configurable overlap
    Time Complexity: O(n) where n = text length
    Space Complexity: O(n) for storing chunks
    
    Trade-offs:
    - Larger chunks: More context but less precision
    - Smaller chunks: More precision but less context
    - More overlap: Better context preservation but more storage
    """
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        """
        Initialize chunking service with size and overlap settings.
        
        Args:
            chunk_size: Number of characters per chunk (default: 512)
            chunk_overlap: Number of overlapping characters (default: 50)
            
        Why configurable?
        - Different document types need different sizes
        - Legal docs: Larger chunks (more context)
        - Short messages: Smaller chunks (more precision)
        
        Raises:
            ValueError: If overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using sliding window.
        
        Algorithm:
        1. Start at position 0
        2. Extract chunk_size characters
        3. Move forward by step_size (chunk_size - overlap)
        4. Repeat until end of text
        
        Example:
            text = "Hello world this is a test"
            chunk_size = 10, overlap = 3
            
            Chunk 1: "Hello worl" [0:10]
            Chunk 2: "rld this i" [7:17]   ← starts at 10-3=7
            Chunk 3: "is is a te" [14:24]  ← starts at 17-3=14
            Chunk 4: "a test"    [21:27]   ← final chunk
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with overlap
            
        Edge Cases:
        - Empty text → returns []
        - Text shorter than chunk_size → returns [text]
        - Last chunk may be smaller than chunk_size
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []
        
        # Clean text (remove extra whitespace)
        text = text.strip()
        
        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Extract chunk from start to start+chunk_size
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            
            chunks.append(chunk)
            
            # Move forward by step_size
            start += self.step_size
            
            # Break if we've reached the end
            # This prevents creating tiny duplicate last chunks
            if end == len(text):
                break
        
        return chunks
    
    def chunk_with_metadata(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Chunk text and attach metadata to each chunk.
        
        Why metadata per chunk?
        - Track source document and page
        - Enable metadata filtering in searches
        - Debug which chunks matched queries
        - Reconstruct original document order
        
        Args:
            text: Input text to chunk
            base_metadata: Metadata to inherit (document_id, page_number, etc.)
            
        Returns:
            List of dicts with 'text' and 'metadata' keys
            
        Example:
            base_metadata = {
                "document_id": "doc123",
                "page_number": 1,
                "source": "ocr"
            }
            
            Returns:
            [
                {
                    "text": "First chunk...",
                    "metadata": {
                        "document_id": "doc123",
                        "page_number": 1,
                        "source": "ocr",
                        "chunk_index": 0,
                        "chunk_size": 512,
                        "start_pos": 0,
                        "end_pos": 512
                    }
                },
                ...
            ]
        """
        chunks = self.chunk_text(text)
        
        chunks_with_metadata = []
        current_pos = 0
        
        for idx, chunk in enumerate(chunks):
            # Create metadata for this chunk
            chunk_metadata = {
                **base_metadata,  # Inherit document-level metadata
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "start_pos": current_pos,
                "end_pos": current_pos + len(chunk)
            }
            
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
            
            # Update position for next chunk
            current_pos += self.step_size
        
        return chunks_with_metadata
    
    def get_stats(self) -> Dict:
        """
        Return chunking configuration statistics.
        
        Why useful?
        - Debugging: What settings are active?
        - Documentation: What was used for this index?
        - Performance analysis: Optimal chunk size testing
        
        Returns:
            Dict with chunk_size, overlap, and effective step_size
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "step_size": self.step_size,
            "overlap_percentage": (self.chunk_overlap / self.chunk_size) * 100
        }
