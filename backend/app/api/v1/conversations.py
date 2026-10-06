"""Conversations API router."""

from typing import List
from fastapi import APIRouter
from backend.app.models.conversation import Conversation
from backend.app.services.conversations.manager import ConversationManager

router = APIRouter(tags=["Conversations"])
conv_manager = ConversationManager()


@router.post("/conversations", response_model=Conversation)
async def create_conversation() -> Conversation:
    """Create a new research conversation session."""
    return conv_manager.create_conversation()
