#!/usr/bin/env python3
"""Run retrieval similarity demo: compare query embedding against chunk embeddings, rank chunks, and print results."""

import asyncio
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.retrieval.demo import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
