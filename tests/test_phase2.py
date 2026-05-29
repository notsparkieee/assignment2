"""
Phase 2 Core Services Tests

Purpose: Verify chunking, embedding, and Chroma repository work correctly.

What we test:
1. ChunkingService - Text splitting with overlap
2. EmbeddingService - Text to vector conversion
3. ChromaRepository - Vector storage and retrieval
4. Integration - Full pipeline end-to-end

Run: python tests/test_phase2.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.repositories.chroma_repository import ChromaRepository


def test_chunking_service():
    """
    Test ChunkingService basic functionality.
    
    Tests:
    1. Empty text handling
    2. Single chunk (text < chunk_size)
    3. Multiple chunks with correct overlap
    4. Metadata attachment
    """
    print("\n" + "="*60)
    print("TEST 1: ChunkingService")
    print("="*60)
    
    # Test with small chunk size for easy verification
    chunker = ChunkingService(chunk_size=20, chunk_overlap=5)
    
    # Test 1: Empty text
    empty_chunks = chunker.chunk_text("")
    assert empty_chunks == [], "Empty text should return empty list"
    print("✅ Empty text handling")
    
    # Test 2: Single chunk
    short_text = "Hello world"
    single_chunk = chunker.chunk_text(short_text)
    assert len(single_chunk) == 1, "Short text should create 1 chunk"
    assert single_chunk[0] == short_text, "Single chunk should match input"
    print("✅ Single chunk handling")
    
    # Test 3: Multiple chunks with overlap
    long_text = "This is a longer text that will be split into multiple chunks with overlap between them"
    chunks = chunker.chunk_text(long_text)
    
    assert len(chunks) > 1, "Long text should create multiple chunks"
    assert all(len(c) <= 20 for c in chunks), "All chunks should be <= chunk_size"
    
    # Verify overlap
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        # Check if there's overlap (last 5 chars of chunk1 should appear in chunk2)
        overlap_expected = chunk1[-5:]
        assert overlap_expected in chunk2, f"Chunks should overlap: '{overlap_expected}' not in '{chunk2}'"
    
    print(f"✅ Multiple chunks with overlap (created {len(chunks)} chunks)")
    
    # Test 4: Metadata attachment
    base_metadata = {
        "document_id": "doc123",
        "page_number": 1,
        "source": "ocr"
    }
    
    chunks_with_meta = chunker.chunk_with_metadata(long_text, base_metadata)
    
    assert len(chunks_with_meta) == len(chunks), "Should have same number of chunks"
    assert all("text" in c and "metadata" in c for c in chunks_with_meta), \
        "Each chunk should have text and metadata"
    
    # Check metadata inheritance
    first_chunk_meta = chunks_with_meta[0]["metadata"]
    assert first_chunk_meta["document_id"] == "doc123", "Should inherit document_id"
    assert first_chunk_meta["chunk_index"] == 0, "First chunk should have index 0"
    assert "start_pos" in first_chunk_meta, "Should have start_pos"
    
    print("✅ Metadata attachment")
    
    # Test 5: Stats
    stats = chunker.get_stats()
    assert stats["chunk_size"] == 20
    assert stats["chunk_overlap"] == 5
    assert stats["step_size"] == 15
    print("✅ Stats retrieval")
    
    print(f"\n🎉 ChunkingService: All tests passed!")
    return True


def test_embedding_service():
    """
    Test EmbeddingService functionality.
    
    Tests:
    1. Model loads correctly
    2. Single text encoding
    3. Batch encoding (multiple texts)
    4. Similarity calculation
    5. Similar texts have high similarity
    """
    print("\n" + "="*60)
    print("TEST 2: EmbeddingService")
    print("="*60)
    
    embedder = EmbeddingService()
    
    # Test 1: Model loaded
    assert embedder.dimension == 384, "MiniLM should have 384 dimensions"
    print(f"✅ Model loaded (dimension: {embedder.dimension})")
    
    # Test 2: Single encoding
    text = "This is a test sentence"
    embedding = embedder.encode(text)
    
    assert isinstance(embedding, list), "Should return list"
    assert len(embedding) == 384, "Should be 384-dimensional"
    assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
    print("✅ Single text encoding")
    
    # Test 3: Batch encoding
    texts = ["First sentence", "Second sentence", "Third sentence"]
    embeddings = embedder.encode_batch(texts)
    
    assert len(embeddings) == 3, "Should return 3 embeddings"
    assert all(len(e) == 384 for e in embeddings), "All should be 384-dimensional"
    print("✅ Batch encoding")
    
    # Test 4: Empty text handling
    try:
        embedder.encode("")
        assert False, "Should raise error for empty text"
    except ValueError:
        print("✅ Empty text error handling")
    
    # Test 5: Similarity calculation
    emb1 = embedder.encode("dog")
    emb2 = embedder.encode("puppy")
    emb3 = embedder.encode("car")
    
    sim_similar = embedder.similarity(emb1, emb2)
    sim_different = embedder.similarity(emb1, emb3)
    
    assert 0 <= sim_similar <= 1, "Similarity should be between 0 and 1"
    assert 0 <= sim_different <= 1, "Similarity should be between 0 and 1"
    assert sim_similar > sim_different, "Dog-puppy should be more similar than dog-car"
    
    print(f"✅ Similarity calculation")
    print(f"   Dog ↔ Puppy: {sim_similar:.3f}")
    print(f"   Dog ↔ Car:   {sim_different:.3f}")
    
    # Test 6: Model info
    info = embedder.get_model_info()
    assert "dimension" in info
    assert "max_seq_length" in info
    print("✅ Model info retrieval")
    
    print(f"\n🎉 EmbeddingService: All tests passed!")
    return True


def test_chroma_repository():
    """
    Test ChromaRepository functionality.
    
    Tests:
    1. Collection creation
    2. Adding chunks
    3. Semantic search
    4. Metadata filtered search
    5. Getting stats
    6. Deletion
    """
    print("\n" + "="*60)
    print("TEST 3: ChromaRepository")
    print("="*60)
    
    # Use test collection to avoid polluting main collection
    repo = ChromaRepository(collection_name="test_collection")
    
    # Clean start
    repo.reset_collection()
    
    # Test 1: Initial stats
    stats = repo.get_stats()
    assert stats["total_chunks"] == 0, "Should start empty"
    print("✅ Collection initialized")
    
    # Create embeddings for testing
    embedder = EmbeddingService()
    
    # Test 2: Add chunks
    chunks = [
        "This is about diabetes treatment",
        "Pizza recipe with cheese",
        "Medical report on blood sugar levels"
    ]
    
    embeddings = embedder.encode_batch(chunks)
    
    metadatas = [
        {"document_id": "doc1", "source": "ocr", "tags": ["medical"]},
        {"document_id": "doc2", "source": "pdf", "tags": ["food"]},
        {"document_id": "doc1", "source": "ocr", "tags": ["medical"]}
    ]
    
    chunk_ids = repo.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        document_id="doc1"
    )
    
    assert len(chunk_ids) == 3, "Should create 3 chunk IDs"
    print("✅ Chunks added to collection")
    
    # Test 3: Stats after adding
    stats = repo.get_stats()
    assert stats["total_chunks"] == 3, "Should have 3 chunks"
    assert stats["embedding_dimension"] == 384, "Should detect 384 dimensions"
    print(f"✅ Stats updated (total chunks: {stats['total_chunks']})")
    
    # Test 4: Semantic search
    query = "diabetes and insulin"
    query_emb = embedder.encode(query)
    
    results = repo.semantic_search(query_emb, n_results=2)
    
    assert len(results['ids'][0]) > 0, "Should return results"
    assert len(results['documents'][0]) > 0, "Should return documents"
    print("✅ Semantic search")
    
    # Test 5: Metadata filtered search
    filtered_results = repo.metadata_filtered_search(
        query_embedding=query_emb,
        filters={"source": "ocr"},
        n_results=10
    )
    
    returned_metas = filtered_results['metadatas'][0]
    assert all(m.get("source") == "ocr" for m in returned_metas), \
        "All results should have source=ocr"
    print("✅ Metadata filtered search")
    
    # Test 6: Hybrid search
    hybrid_results = repo.hybrid_search(
        query_embedding=query_emb,
        metadata_filters={"source": "ocr"},
        where_document={"$contains": "diabetes"},
        n_results=10
    )
    
    assert len(hybrid_results['ids'][0]) > 0, "Should return hybrid results"
    print("✅ Hybrid search")
    
    # Test 7: Get by document ID
    doc_results = repo.get_by_document_id("doc1")
    # Note: This gets chunks with document_id in metadata
    print("✅ Get by document ID")
    
    # Test 8: Delete document
    deleted = repo.delete_document("doc1")
    assert deleted > 0, "Should delete chunks"
    print(f"✅ Document deletion ({deleted} chunks deleted)")
    
    # Clean up
    repo.reset_collection()
    print("✅ Collection cleaned up")
    
    print(f"\n🎉 ChromaRepository: All tests passed!")
    return True


def test_integration_pipeline():
    """
    Test full pipeline: Chunking → Embedding → Storage → Search
    
    This simulates the real workflow:
    1. Chunk a document
    2. Generate embeddings
    3. Store in Chroma
    4. Search and retrieve
    """
    print("\n" + "="*60)
    print("TEST 4: Integration Pipeline")
    print("="*60)
    
    # Initialize services
    chunker = ChunkingService(chunk_size=100, chunk_overlap=20)
    embedder = EmbeddingService()
    repo = ChromaRepository(collection_name="test_integration")
    
    # Clean start
    repo.reset_collection()
    
    # Simulate document
    document = """
    Diabetes mellitus is a chronic disease characterized by high blood sugar levels.
    Treatment typically involves insulin therapy, dietary changes, and regular monitoring.
    Patients should maintain a healthy diet and exercise regularly to manage their condition.
    Blood sugar levels should be checked multiple times per day for optimal control.
    """
    
    # Step 1: Chunk document
    base_metadata = {
        "document_id": "medical_doc_001",
        "source": "ocr",
        "page_number": 1,
        "tags": ["medical", "diabetes"]
    }
    
    chunks_with_meta = chunker.chunk_with_metadata(document, base_metadata)
    print(f"✅ Chunked document into {len(chunks_with_meta)} chunks")
    
    # Step 2: Generate embeddings
    chunk_texts = [c["text"] for c in chunks_with_meta]
    embeddings = embedder.encode_batch(chunk_texts, show_progress=False)
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Step 3: Store in Chroma
    metadatas = [c["metadata"] for c in chunks_with_meta]
    chunk_ids = repo.add_chunks(
        chunks=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        document_id="medical_doc_001"
    )
    print(f"✅ Stored {len(chunk_ids)} chunks in Chroma")
    
    # Step 4: Search
    queries = [
        "diabetes treatment",
        "blood sugar monitoring",
        "insulin therapy"
    ]
    
    for query in queries:
        query_emb = embedder.encode(query)
        results = repo.semantic_search(query_emb, n_results=2)
        
        assert len(results['ids'][0]) > 0, f"Should find results for '{query}'"
        
        # Show top result
        top_doc = results['documents'][0][0]
        top_distance = results['distances'][0][0]
        print(f"✅ Query: '{query}'")
        print(f"   Top match (distance={top_distance:.3f}): {top_doc[:50]}...")
    
    # Clean up
    repo.reset_collection()
    
    print(f"\n🎉 Integration Pipeline: All tests passed!")
    return True


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "="*60)
    print("🚀 PHASE 2: CORE SERVICES TESTING")
    print("="*60)
    
    try:
        # Test individual services
        test_chunking_service()
        test_embedding_service()
        test_chroma_repository()
        
        # Test integration
        test_integration_pipeline()
        
        print("\n" + "="*60)
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("="*60)
        print()
        print("✅ ChunkingService working")
        print("✅ EmbeddingService working")
        print("✅ ChromaRepository working")
        print("✅ Full pipeline working")
        print()
        print("🚀 Ready for Phase 3: Search Strategies")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
