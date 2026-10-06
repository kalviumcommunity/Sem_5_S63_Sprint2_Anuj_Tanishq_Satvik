"""Integration tests for health check and root endpoints."""

import pytest


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ResearchMate"
    assert data["status"] == "online"
    assert "principle" in data


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "ResearchMate"
    assert "api" in data["services"]


@pytest.mark.asyncio
async def test_api_v1_health_check(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["environment"] == "development"
