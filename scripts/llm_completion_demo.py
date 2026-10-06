"""Demo script for testing LLM completion calls."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure stdout handles utf-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings
from backend.app.services.llm import (
    CompletionResponse,
    LLMConfig,
    LLMException,
    get_llm_client,
)
from backend.app.services.llm.prompts import ACADEMIC_RAG_SYSTEM_PROMPT


async def run_demo(
    prompt: str,
    provider: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
):
    """Execute a controlled LLM completion call and display results."""
    target_provider = provider or settings.LLM_PROVIDER
    target_model = model or settings.LLM_MODEL
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

    print("\n" + "=" * 65)
    print("ResearchMate LLM Completion Demo")
    print("=" * 65)
    print(f"Provider:    {target_provider}")
    print(f"Model:       {target_model}")
    print(f"Temperature: {temp}")
    print(f"Max Tokens:  {tokens}")
    print(f"Prompt:      {prompt}")
    print("-" * 65)

    config = LLMConfig(
        model=target_model,
        temperature=temp,
        max_tokens=tokens,
        timeout_seconds=30.0,
    )
    client = get_llm_client(provider=target_provider, config=config)

    try:
        response: CompletionResponse = await client.complete(
            prompt=prompt,
            system_prompt=ACADEMIC_RAG_SYSTEM_PROMPT,
        )

        print("\n[SUCCESS] Completion Generated:")
        print(f"Response Model:     {response.model}")
        print(f"Provider:           {response.provider}")
        print(f"Prompt Tokens:      {response.prompt_tokens}")
        print(f"Completion Tokens:  {response.completion_tokens}")
        print(f"Total Tokens:       {response.total_tokens}")
        print(f"Estimated Cost:     ${response.estimated_cost_usd:.6f} USD")
        print(f"Finish Reason:      {response.finish_reason}")
        print(f"Latency:            {response.latency_ms:.2f} ms")
        print("\nOutput Text:\n")
        print(response.text)
        print("\n" + "=" * 65 + "\n")
        return response

    except LLMException as err:
        print(f"\n[ERROR] LLM Exception: {err}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ResearchMate LLM Completion Demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the primary objective of federated learning in privacy-preserving machine learning?",
        help="Input research question or prompt",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider: mock, openai, gemini (defaults to LLM_PROVIDER in .env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., gpt-4o-mini, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature parameter (0.0 to 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()
    asyncio.run(
        run_demo(
            prompt=args.prompt,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
