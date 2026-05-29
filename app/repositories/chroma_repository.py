"""
ChromaRepository: Vector database operations using Chroma

Purpose: Store and retrieve document embeddings with metadata for semantic search.

What is Chroma?
- Open-source vector database
- Built-in persistence (data survives restarts)
- Native metadata filtering
- HNSW index for fast similarity search

Example:
    repo = ChromaRepository()
    repo.add_chunks(chunks=["text1"], embeddings=[[0.1, ...]], metadatas=[{...}])
    results = repo.semantic_search(query_embedding, n_results=10)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from app.config import settings


class ChromaRepository:
    """
    Repository for vector database operations using Chroma.
    
    Why Chroma over FAISS?
    - Native metadata filtering (no manual coding)
    - Automatic persistence (data survives restarts)
    - Simple updates/deletes (FAISS requires rebuild)
    - Production-ready (auth, multi-tenancy)
    - Python-first API (easier to use)
    
    Index Type: HNSW (Hierarchical Navigable Small World)
    - Time Complexity: O(log n) for search
    - Space Complexity: O(n·d) where d = embedding dimension
    - Trade-off: Approximate but very fast
    
    Why Repository Pattern?
    - Encapsulates all Chroma operations
    - Easy to test (can mock)
    - Can swap vector DB later without changing service layer
    - Clean separation of concerns
    """
    
    def __init__(
        self,
        persist_directory: str = settings.CHROMA_PERSIST_DIRECTORY,
        collection_name: str = settings.CHROMA_COLLECTION_NAME
    ):
        """
        Initialize Chroma client and collection.
        
        Why PersistentClient?
        - Data survives restarts (production requirement)
        - No re-indexing needed after restart
        - Fast startup (loads existing index)
        
        Alternative: EphemeralClient (in-memory, testing only)
        
        Args:
            persist_directory: Path to store Chroma data
            collection_name: Name of the collection
        """
        print(f"🔄 Initializing Chroma repository...")
        print(f"   Persist directory: {persist_directory}")
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable analytics in production
                allow_reset=True  # Allow collection reset (useful for development)
            )
        )
        
        # Get or create collection
        # Why get_or_create?
        # - First run: creates collection
        # - Subsequent runs: loads existing data
        # - No error if collection exists
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # Distance function for similarity
                "description": "Document chunks with embeddings"
            }
        )
        
        print(f"✅ Chroma repository initialized")
        print(f"   Collection: {collection_name}")
        print(f"   Existing chunks: {self.collection.count()}")
    
    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        document_id: str
    ) -> List[str]:
        """
        Add document chunks to Chroma collection.
        
        Why batch insert?
        - Faster than one-by-one (reduces overhead)
        - Atomic operation (all or nothing)
        - Single transaction to database
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            document_id: Base document ID
            
        Returns:
            List of chunk IDs generated
            
        Why generate IDs?
        - Chroma requires unique IDs
        - Format: {document_id}_chunk_{index}
        - Easy to query by document
        
        Example:
            add_chunks(
                chunks=["chunk1", "chunk2"],
                embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
                metadatas=[{"page": 1}, {"page": 1}],
                document_id="doc123"
            )
            → Returns: ["doc123_chunk_0", "doc123_chunk_1"]
        """
        if not chunks or not embeddings or not metadatas:
            raise ValueError("chunks, embeddings, and metadatas cannot be empty")
        
        if not (len(chunks) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Length mismatch: chunks={len(chunks)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        
        # Generate unique IDs for each chunk
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Add to Chroma collection
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,        # Original text
            embeddings=embeddings,   # Vector representation
            metadatas=metadatas      # Filterable metadata
        )
        
        print(f"✅ Added {len(chunk_ids)} chunks for document: {document_id}")
        
        return chunk_ids
    
    def semantic_search(
        self,
        query_embedding: List[float],
        n_results: int = 10
    ) -> Dict:
        """
        Pure vector similarity search (no metadata filtering).
        
        How it works:
        1. Calculate cosine similarity between query and all vectors
        2. Return top-k most similar
        
        Time Complexity: O(log n) with HNSW index
        
        Why fast?
        - HNSW creates navigable graph structure
        - Doesn't compare with every vector
        - Approximate nearest neighbors (99%+ accuracy)
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
            
        Example:
            results = semantic_search([0.1, 0.2, ...], n_results=10)
            {
                'ids': [['doc1_chunk_0', 'doc2_chunk_3']],
                'documents': [['text1', 'text2']],
                'metadatas': [[{...}, {...}]],
                'distances': [[0.15, 0.23]]  # Lower = more similar
            }
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def metadata_filtered_search(
        self,
        query_embedding: List[float],
        filters: Dict,
        n_results: int = 10
    ) -> Dict:
        """
        Vector search with metadata filtering.
        
        Why filter first?
        - Reduces search space (faster)
        - Example: 1M docs → filter to 10K medical → 100x faster
        - Still uses HNSW for vector similarity
        
        Time Complexity: O(k·log k) where k = filtered subset size
        
        Args:
            query_embedding: Query vector
            filters: Metadata filter conditions
            n_results: Number of results
            
        Filter syntax (Chroma):
            {"source": "ocr"}  # Exact match
            {"tags": {"$in": ["medical", "invoice"]}}  # Contains any
            {"page_number": {"$gte": 5, "$lte": 10}}  # Range
            {"$and": [{...}, {...}]}  # Multiple conditions
            
        Example:
            results = metadata_filtered_search(
                query_embedding=[0.1, ...],
                filters={"source": "ocr", "tags": {"$in": ["medical"]}},
                n_results=10
            )
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=filters,  # Native Chroma filtering
            n_results=n_results
        )
        
        return results
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        metadata_filters: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        n_results: int = 10
    ) -> Dict:
        """
        Hybrid search: Vector + Metadata + Keyword filtering.
        
        Most powerful search mode - combines:
        1. Vector similarity (semantic meaning)
        2. Metadata filters (structured data)
        3. Keyword matching (exact terms)
        
        Use cases:
        - "Find medical invoices about diabetes from 2024"
          * Vector: semantic match on "diabetes", "blood sugar"
          * Metadata: source="ocr", tags=["medical"], year=2024
          * Keywords: ensure "invoice" appears in text
        
        Args:
            query_embedding: Query vector
            metadata_filters: Metadata conditions
            where_document: Keyword matching on document text
            n_results: Number of results
            
        where_document syntax:
            {"$contains": "invoice"}  # Must contain word
            {"$or": [{"$contains": "diabetes"}, {"$contains": "insulin"}]}
            
        Example:
            results = hybrid_search(
                query_embedding=[0.1, ...],
                metadata_filters={"source": "ocr"},
                where_document={"$contains": "invoice"},
                n_results=10
            )
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=metadata_filters,        # Metadata filtering
            where_document=where_document,  # Keyword matching
            n_results=n_results
        )
        
        return results
    
    def get_by_document_id(self, document_id: str) -> Dict:
        """
        Get all chunks for a specific document.
        
        Why useful?
        - Retrieve original document
        - Update document (delete old, add new)
        - GDPR compliance (delete user data)
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with all chunks for document
        """
        # Query by ID prefix pattern
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        return results
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Why needed in production?
        - User deletes document
        - GDPR "right to be forgotten"
        - Update workflow (delete old, add new)
        
        Args:
            document_id: Document to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get all chunks for this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if results['ids']:
            # Delete by IDs
            self.collection.delete(ids=results['ids'])
            deleted_count = len(results['ids'])
            print(f"✅ Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
        
        return 0
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Why needed?
        - Production monitoring (how much data stored?)
        - Debugging (is data being indexed?)
        - Cost estimation (storage requirements)
        
        Returns:
            Dict with total chunks, collection name, embedding dimension
        """
        total_chunks = self.collection.count()
        
        # Get embedding dimension from first vector (if present)
        embedding_dim = None
        if total_chunks > 0:
            sample = self.collection.peek(limit=1)
            embeddings = sample.get("embeddings")
            if (
                embeddings is not None
                and len(embeddings) > 0
                and embeddings[0] is not None
            ):
                embedding_dim = len(embeddings[0])
        if embedding_dim is None:
            embedding_dim = settings.EMBEDDING_DIMENSION
        
        return {
            "total_chunks": total_chunks,
            "collection_name": self.collection.name,
            "embedding_dimension": embedding_dim,
            "persist_directory": settings.CHROMA_PERSIST_DIRECTORY
        }
    
    def reset_collection(self):
        """
        Delete all data in collection (use with caution!).
        
        Why needed?
        - Development/testing
        - Re-indexing from scratch
        - Fixing data corruption
        
        WARNING: This is destructive!
        """
        name = self.collection.name
        metadata = dict(self.collection.metadata or {})
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata,
        )
        print("Collection reset: all data deleted")
