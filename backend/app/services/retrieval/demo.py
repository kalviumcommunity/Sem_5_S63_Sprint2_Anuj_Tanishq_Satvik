"""Demonstration of query embedding, cosine similarity calculation, and chunk ranking."""

import asyncio

from backend.app.core.config import settings
from backend.app.services.embeddings.service import get_embedding_service
from backend.app.services.retrieval.search import cosine_similarity, rank_chunks


async def main() -> None:
    query = "How can a learner reset their password?"
    service = get_embedding_service(use_cache=False)

    # 1. Embed the query using the same model used for document chunks
    query_embeddings = await service.embed_batch([query])
    query_embedding = query_embeddings[0]

    # 2. Document chunk records with metadata
    raw_chunks = [
        {
            "text": "Password reset instructions for learner accounts.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0},
        },
        {
            "text": "The cafeteria menu changes every Friday.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 3},
        },
        {
            "text": "Learners can recover access using their registered email.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1},
        },
    ]

    # 3. Generate embeddings for stored chunks
    chunk_texts = [item["text"] for item in raw_chunks]
    chunk_embeddings = await service.embed_batch(chunk_texts)

    chunk_records = [
        {
            "text": item["text"],
            "metadata": item["metadata"],
            "embedding": emb,
        }
        for item, emb in zip(raw_chunks, chunk_embeddings)
    ]

    # 4. Score query embedding against every chunk and rank from most to least similar
    ranked = rank_chunks(query_embedding, chunk_records)

    has_api_key = bool(
        settings.EMBEDDING_API_KEY
        and not settings.EMBEDDING_API_KEY.startswith("your-")
        and "placeholder" not in settings.EMBEDDING_API_KEY
    )
    mode = (
        "configured OpenAI semantic embeddings"
        if settings.EMBEDDING_PROVIDER.lower() == "openai" and has_api_key
        else "offline lexical mock"
    )

    print(f"mode: {mode}")
    print("query:", query)
    print("most similar:", round(ranked[0]["score"], 4), ranked[0]["text"], ranked[0]["metadata"])
    print("least similar:", round(ranked[-1]["score"], 4), ranked[-1]["text"], ranked[-1]["metadata"])


if __name__ == "__main__":
    asyncio.run(main())
