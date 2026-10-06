"""Core application configuration using Pydantic Settings."""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "ResearchMate"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
    ]

    # Security
    SECRET_KEY: str = "default-secret-key-change-in-production-min-32-chars"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # LLM Settings
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 32

    # Vector Database
    VECTOR_DB_TYPE: str = "memory"
    VECTOR_DB_PATH: str = "./data/vector_db"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"

    # Document Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_FILE_EXTENSIONS: Union[List[str], str] = [".pdf", ".docx", ".txt", ".md"]
    UPLOAD_DIRECTORY: str = "./data/uploads"

    # Retrieval & Guardrails
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANKED: int = 5
    SIMILARITY_THRESHOLD: float = 0.65
    GROUNDING_THRESHOLD: float = 0.60
    REFUSAL_MESSAGE: str = (
        "I couldn't find enough supporting information in the available academic sources to answer this reliably."
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @field_validator("ALLOWED_FILE_EXTENSIONS", mode="before")
    @classmethod
    def assemble_allowed_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


settings = Settings()
