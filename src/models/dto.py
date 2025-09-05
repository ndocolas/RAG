from pydantic import BaseModel, Field
from typing import List, Optional, Dict


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
    question: str
    top_k: int = 5
    mmr_lambda: float = 0.5


class Citation(BaseModel):
    source_id: str
    page: int
    score: float
    snippet: str


class AIResponse(BaseModel):
    session_id: str
    user_input: str
    response: str
    citations: List[Citation] = []
    metadata: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="latency_ms, tokens_in, tokens_out, etc."
    )
