"""Conversation and Message data models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Represents a validated message in a conversation thread."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(..., description="Unique conversation identifier")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Text content of the message")
    token_count: Optional[int] = Field(default=None, description="Pre-computed token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom message metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("conversation_id cannot be empty or whitespace")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if v is None:
            raise ValueError("content cannot be None")
        return v


class Conversation(BaseModel):
    """Represents an isolated conversation session for multi-turn research queries."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    title: str = "New Research Inquiry"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
