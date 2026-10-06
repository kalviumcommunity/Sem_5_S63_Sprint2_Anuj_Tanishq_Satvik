"""Repository for conversations and messages."""

from typing import Dict, List, Optional
from backend.app.models.conversation import Conversation, Message


class ConversationRepository:
    """In-memory repository for conversations and messages."""

    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[Message]] = {}

    def save_conversation(self, conv: Conversation) -> Conversation:
        self._conversations[conv.id] = conv
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self._conversations.get(conv_id)

    def save_message(self, message: Message) -> Message:
        self._messages.setdefault(message.conversation_id, []).append(message)
        return message

    def get_messages(self, conv_id: str) -> List[Message]:
        return self._messages.get(conv_id, [])
