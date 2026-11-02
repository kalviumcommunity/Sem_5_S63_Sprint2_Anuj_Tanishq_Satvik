"""Print dimensions and cosine similarities for a small academic-text example."""

import asyncio

from backend.app.core.config import settings
from backend.app.services.embeddings.service import get_embedding_service
from backend.app.services.retrieval.search import cosine_similarity


async def main() -> None:
    texts = [
        "Neural networks learn patterns from academic training data.",
        "Neural network models learn patterns from research training data.",
        "Medieval trade routes connected cities through maritime commerce.",
    ]
    service = get_embedding_service(use_cache=False)
    vectors = await service.embed_batch(texts)
    has_api_key = bool(
        settings.EMBEDDING_API_KEY
        and not settings.EMBEDDING_API_KEY.startswith("your-")
        and "placeholder" not in settings.EMBEDDING_API_KEY
    )
    mode = (
        "configured OpenAI semantic embeddings"
        if settings.EMBEDDING_PROVIDER.lower() == "openai" and has_api_key
        else "offline lexical mock (not semantic)"
    )

    print(f"mode: {mode}")
    print(f"vector dimensions: {[len(vector) for vector in vectors]}")
    print(f"related cosine similarity: {cosine_similarity(vectors[0], vectors[1]):.4f}")
    print(f"unrelated cosine similarity: {cosine_similarity(vectors[0], vectors[2]):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
