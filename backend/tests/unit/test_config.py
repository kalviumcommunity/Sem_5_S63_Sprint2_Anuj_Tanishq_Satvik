"""Unit tests for configuration loading."""

from backend.app.core.config import Settings


def test_settings_default_values():
    settings = Settings()
    assert settings.APP_NAME == "ResearchMate"
    assert settings.DEBUG is True
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.TOP_K_RETRIEVAL == 10
    assert settings.CHUNK_SIZE == 500
    assert settings.CHUNK_OVERLAP == 100
    assert "No unsupported answer" in settings.REFUSAL_MESSAGE or "supporting information" in settings.REFUSAL_MESSAGE


def test_settings_cors_origins_parsing():
    settings = Settings(CORS_ORIGINS="http://localhost:3000,http://example.com")
    assert "http://localhost:3000" in settings.CORS_ORIGINS
    assert "http://example.com" in settings.CORS_ORIGINS
