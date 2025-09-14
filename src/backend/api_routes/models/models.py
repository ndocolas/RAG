"""Data models for API request and response payloads."""

from pydantic import BaseModel
from src.backend.services.chat_service.models.models import AIChatOutput


class UserRequest(BaseModel):
    """Model for user request payload."""

    session_id: str
    user_input: str


class ChatResponse(BaseModel):
    """Represents the output of a chat interaction."""

    session_id: str
    user_input: str
    response_model: AIChatOutput
