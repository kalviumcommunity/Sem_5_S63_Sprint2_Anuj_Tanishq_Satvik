"""Conversation management and in-memory session tracking."""

from typing import Dict, List, Optional
from backend.app.models.conversation import Conversation, Message, MessageRole


class ConversationManager:
    """Manages conversations and associated message history."""

    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[Message]] = {}

    def create_conversation(self, title: str = "New Inquiry") -> Conversation:
        conv = Conversation(title=title)
        self._conversations[conv.id] = conv
        self._messages[conv.id] = []
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self._conversations.get(conv_id)

    def add_message(self, conv_id: str, role: MessageRole, content: str) -> Message:
        if conv_id not in self._conversations:
            self.create_conversation()
        msg = Message(conversation_id=conv_id, role=role, content=content)
        self._messages.setdefault(conv_id, []).append(msg)
        return msg

    def get_history(self, conv_id: Optional[str]) -> List[Message]:
        if not conv_id:
            return []
        return self._messages.get(conv_id, [])
