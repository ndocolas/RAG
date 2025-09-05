from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uuid
from src.models.dto import StartChatResponse, QuestionRequest, AIResponse
from src.services.rag import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()


@router.post("/start_chat", response_model=StartChatResponse)
def start_chat():
    return StartChatResponse(session_id=str(uuid.uuid4()))


@router.post("/ask", response_model=AIResponse)
def ask(req: QuestionRequest):
    out, cites, elapsed = pipeline.ask(
        session_id=req.session_id, question=req.question,
        top_k=req.top_k, mmr_lambda=req.mmr_lambda, stream=False
    )
    return AIResponse(
        session_id=req.session_id,
        user_input=req.question,
        response=out,
        citations=cites,
        metadata={"latency_ms": elapsed * 1000}
    )


@router.post("/ask/stream")
def ask_stream(req: QuestionRequest):
    def gen():
        stream, cites, _ = pipeline.ask(
            session_id=req.session_id, question=req.question,
            top_k=req.top_k, mmr_lambda=req.mmr_lambda, stream=True
        )
        yield {"event": "citations", "data": [c.model_dump() for c in cites]}

        for chunk in stream:
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(gen())
