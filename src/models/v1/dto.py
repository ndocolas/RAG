from pydantic import BaseModel
from typing import List


class StartChatResponse(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    message: str
    session_id: str
    docs_indexed: int
    chunks: int
    points: int
    sources: List[str] = []


class QuestionRequest(BaseModel):
    session_id: str
    user_input: str


class Citation(BaseModel):
    source_id: str
    page: int
    score: float
    snippet: str


class AIResponse(BaseModel):
    session_id: str
    user_input: str
    response: str
