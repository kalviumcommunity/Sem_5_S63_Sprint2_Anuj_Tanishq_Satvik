"""Unit tests for Concept #12: Embeddings Generation & Batching."""

import math
import pytest
from backend.app.services.embeddings.batcher import EmbeddingBatcher, chunk_list
from backend.app.services.embeddings.cache import EmbeddingCache
from backend.app.services.embeddings.service import (
    CachedEmbeddingService,
    EmbeddingServiceError,
    MockEmbeddingService,
    OpenAIEmbeddingService,
    get_embedding_service,
    validate_embedding,
)
from backend.app.services.retrieval.search import cosine_similarity


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
async def test_academic_embeddings_rank_related_text_above_unrelated_text():
    """The offline lexical mock retains shared terms without claiming semantic inference."""
    svc = MockEmbeddingService(dimension=512)
    vectors = await svc.embed_batch(
        [
            "Neural networks learn patterns from academic training data.",
            "Neural network models learn patterns from research training data.",
            "Medieval trade routes connected cities through maritime commerce.",
        ]
    )

    related_similarity = cosine_similarity(vectors[0], vectors[1])
    unrelated_similarity = cosine_similarity(vectors[0], vectors[2])
    assert related_similarity > unrelated_similarity


def test_embedding_validation_checks_dimension_and_finite_values():
    validate_embedding([0.6, 0.8], expected_dimension=2)
    with pytest.raises(ValueError, match="does not match configured dimension"):
        validate_embedding([1.0], expected_dimension=2)
    with pytest.raises(ValueError, match="finite"):
        validate_embedding([float("nan")], expected_dimension=1)
    assert MockEmbeddingService.normalize_vector([1e308, 1e308]) == pytest.approx(
        [2**-0.5, 2**-0.5]
    )


@pytest.mark.asyncio
async def test_openai_embedding_sends_auth_and_preserves_input_order(monkeypatch):
    observed = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 4.0, 0.0]},
                ]
            }

    class AsyncClient:
        def __init__(self, timeout):
            observed["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, headers, json):
            observed["headers"] = headers
            observed["payload"] = json
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", AsyncClient)
    service = OpenAIEmbeddingService(api_key="test-key", model="test-model", dimension=3)
    vectors = await service.embed_batch(["first", "second"])

    assert observed["headers"]["Authorization"] == "Bearer test-key"
    assert observed["payload"]["input"] == ["first", "second"]
    assert vectors == [[0.6, 0.8, 0.0], [0.0, 0.6, 0.8]]


@pytest.mark.asyncio
async def test_openai_embedding_splits_batches_and_preserves_input_order(monkeypatch):
    observed_batches = []

    class Response:
        status_code = 200

        def __init__(self, texts):
            self.texts = texts

        def json(self):
            return {
                "data": [
                    {
                        "index": index,
                        "embedding": {
                            "first": [1.0, 0.0],
                            "second": [0.0, 1.0],
                            "third": [-1.0, 0.0],
                        }[self.texts[index]],
                    }
                    for index in reversed(range(len(self.texts)))
                ]
            }

    class AsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, headers, json):
            observed_batches.append(json["input"])
            return Response(json["input"])

    monkeypatch.setattr("httpx.AsyncClient", AsyncClient)
    service = OpenAIEmbeddingService(
        api_key="test-key",
        model="test-model",
        dimension=2,
        batch_size=2,
    )
    vectors = await service.embed_batch(["first", "second", "third"])

    assert observed_batches == [["first", "second"], ["third"]]
    assert vectors == [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_data",
    [
        [],
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 0, "embedding": [1.0, float("nan")]}],
        [{"index": 1, "embedding": [1.0, 0.0]}],
    ],
)
async def test_openai_embedding_rejects_invalid_responses(monkeypatch, response_data):
    class Response:
        status_code = 200

        def json(self):
            return {"data": response_data}

    class AsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, headers, json):
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", AsyncClient)
    service = OpenAIEmbeddingService(api_key="test-secret", model="test-model", dimension=2)

    with pytest.raises(EmbeddingServiceError):
        await service.embed_batch(["input"])


@pytest.mark.asyncio
async def test_openai_embedding_errors_do_not_expose_provider_body_or_key(monkeypatch):
    class Response:
        status_code = 401

        def json(self):
            return {"error": "test-secret is invalid"}

    class AsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def post(self, url, headers, json):
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", AsyncClient)
    service = OpenAIEmbeddingService(api_key="test-secret", model="test-model", dimension=2)

    with pytest.raises(EmbeddingServiceError) as error:
        await service.embed_text("input")

    assert "test-secret" not in str(error.value)
    assert "invalid" not in str(error.value)


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


@pytest.mark.asyncio
async def test_cached_embeddings_are_partitioned_by_model():
    class ModelEmbeddingService(MockEmbeddingService):
        def __init__(self, model, vector):
            super().__init__(dimension=2)
            self.model = model
            self.vector = vector

        async def embed_text(self, text):
            return self.vector

        async def embed_batch(self, texts):
            return [self.vector for _ in texts]

    cache = EmbeddingCache(max_size=10)
    service_a = CachedEmbeddingService(ModelEmbeddingService("model-a", [1.0, 0.0]), cache)
    service_b = CachedEmbeddingService(ModelEmbeddingService("model-b", [0.0, 1.0]), cache)

    assert await service_a.embed_text("same input") == [1.0, 0.0]
    assert await service_b.embed_text("same input") == [0.0, 1.0]


def test_get_embedding_service_factory():
    """Verify get_embedding_service returns configured service wrapped with cache."""
    svc = get_embedding_service(use_cache=True)
    assert isinstance(svc, CachedEmbeddingService)
    assert svc.dimension > 0
