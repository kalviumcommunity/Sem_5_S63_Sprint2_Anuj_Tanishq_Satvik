"""Test configuration and shared fixtures."""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def async_client():
    """Async test client using httpx ASGITransport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
