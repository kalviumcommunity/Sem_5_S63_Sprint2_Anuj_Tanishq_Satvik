"""Core module containing application configuration, logging, and security."""

from backend.app.core.config import settings
from backend.app.core.logging import logger

__all__ = ["settings", "logger"]
