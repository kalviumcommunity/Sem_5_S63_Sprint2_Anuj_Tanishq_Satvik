"""Unit tests for Concept #12: Embeddings Generation & Batching."""

import math
import pytest
from backend.app.services.embeddings.batcher import EmbeddingBatcher, chunk_list
from backend.app.services.embeddings.cache import EmbeddingCache
from backend.app.services.embeddings.service import (
    CachedEmbeddingService,
    MockEmbeddingService,
    get_embedding_service,
)


@pytest.mark.asyncio
async def test_mock_embedding_service_dimensions_and_normalization():
    """Verify mock embedding service generates vectors of requested dimension with L2 norm of 1.0."""
    for dim in [128, 384, 768, 1536]:
        svc = MockEmbeddingService(dimension=dim)
        assert svc.dimension == dim

        vec = await svc.embed_text("Deep neural networks in bioinformatics.")
        assert len(vec) == dim

        # Check L2 norm is ~1.0
        norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)


@pytest.mark.asyncio
async def test_mock_embedding_deterministic():
    """Verify that identical text produces identical vectors and distinct text produces distinct vectors."""
    svc = MockEmbeddingService(dimension=256)
    t1 = "Reinforcement learning from human feedback."
    t2 = "Reinforcement learning from human feedback."
    t3 = "A completely different sentence on algebraic topology."

    v1 = await svc.embed_text(t1)
    v2 = await svc.embed_text(t2)
    v3 = await svc.embed_text(t3)

    assert v1 == v2
    assert v1 != v3


@pytest.mark.asyncio
async def test_embed_batch_handles_multiple_texts():
    """Verify embed_batch returns matching vector counts for arbitrary batch sizes."""
    svc = MockEmbeddingService(dimension=128)
    texts = [f"Academic document chunk number {i}" for i in range(15)]

    vectors = await svc.embed_batch(texts)
    assert len(vectors) == 15
    for vec in vectors:
        assert len(vec) == 128


def test_chunk_list_helper():
    """Verify chunk_list divides items into equal batches with expected remainders."""
    items = list(range(25))
    batches = chunk_list(items, batch_size=10)

    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5

    with pytest.raises(ValueError):
        chunk_list(items, batch_size=0)


@pytest.mark.asyncio
async def test_embedding_batcher_pipeline_execution():
    """Verify EmbeddingBatcher executes pipeline in batches with retries."""
    batcher = EmbeddingBatcher(default_batch_size=4, max_retries=2)
    svc = MockEmbeddingService(dimension=64)
    texts = [f"Text line {i}" for i in range(10)]

    call_count = 0

    async def mock_embed_fn(batch):
        nonlocal call_count
        call_count += 1
        return await svc.embed_batch(batch)

    results = await batcher.execute_batch_pipeline(texts, mock_embed_fn, batch_size=4)

    assert len(results) == 10
    # 10 items in batches of 4 -> 3 calls (4, 4, 2)
    assert call_count == 3


def test_embedding_cache_lru_and_metrics():
    """Verify EmbeddingCache LRU eviction and hit/miss tracking."""
    cache = EmbeddingCache(max_size=3)
    v1 = [0.1, 0.2]
    v2 = [0.3, 0.4]
    v3 = [0.5, 0.6]
    v4 = [0.7, 0.8]

    cache.set("text1", v1)
    cache.set("text2", v2)
    cache.set("text3", v3)
    assert cache.size == 3

    # Hits and misses
    assert cache.get("text1") == v1
    assert cache.hits == 1

    assert cache.get("text_not_there") is None
    assert cache.misses == 1

    # Adding 4th item should evict text2 (since text1 was accessed most recently)
    cache.set("text4", v4)
    assert cache.size == 3
    assert cache.get("text2") is None
    assert cache.get("text1") == v1
    assert cache.get("text4") == v4


@pytest.mark.asyncio
async def test_cached_embedding_service_avoids_recalculation():
    """Verify CachedEmbeddingService uses cache on repeated queries."""
    cache = EmbeddingCache(max_size=100)
    base_svc = MockEmbeddingService(dimension=64)
    cached_svc = CachedEmbeddingService(base_service=base_svc, cache=cache)

    text = "Quantum error correction codes in fault-tolerant processors."

    # First call - cache miss
    v1 = await cached_svc.embed_text(text)
    assert cache.misses == 1
    assert cache.hits == 0

    # Second call - cache hit
    v2 = await cached_svc.embed_text(text)
    assert cache.hits == 1
    assert v1 == v2

    # Batch call with mixed cached and uncached texts
    t_new = "A brand new uncached query."
    batch_results = await cached_svc.embed_batch([text, t_new])
    assert len(batch_results) == 2
    assert batch_results[0] == v1
    assert cache.hits == 2


def test_get_embedding_service_factory():
    """Verify get_embedding_service returns configured service wrapped with cache."""
    svc = get_embedding_service(use_cache=True)
    assert isinstance(svc, CachedEmbeddingService)
    assert svc.dimension > 0
