"""Schemas for multi-turn chat interactions and conversational memory."""
from enum import Enum
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Roles in conversational interaction."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Single turn in a conversational history."""
    role: ChatRole = Field(description="Role of the message sender: user, assistant, or system")
    content: str = Field(min_length=1, description="Message text content")

