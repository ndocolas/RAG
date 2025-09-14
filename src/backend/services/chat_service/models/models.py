"""Models for chat service interactions."""

from pydantic import BaseModel, Field
from typing import Optional


class AIChatOutput(BaseModel):
    """Represents the AI's response to a user query in a chat session."""

    response: str = Field(
        description="""
The response that addresses the user question. Base your answer totally on the retrieved chunk.
If there is no answer to the question, retrieve only: I'm sorry, but I don't have enough
information to answer that question.
"""
    )
    reference: Optional[str] = Field(
        description="""The source to explain from where the response was gathered.
Fill with a sentence from the recieved chunk.
If there is no answer to the question, retrieve only: No source was found related to the question."
"""
    )


class ChatOutput(BaseModel):
    """Represents the output of a chat interaction."""

    session_id: str
    user_input: str
    response_model: AIChatOutput
