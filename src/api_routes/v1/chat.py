import uuid
import logging

from fastapi import APIRouter
from src.models.v1.dto import StartChatResponse, QuestionRequest, AIResponse
from src.services.chat_service import ChatService
from src.adapters.llm_gemini import GeminiClient

logger = logging.getLogger(__name__)

router = APIRouter()

llm = GeminiClient()
chat_service = ChatService(llm)

@router.post("/start_chat", response_model=StartChatResponse)
def start_chat():
    return StartChatResponse(session_id=str(uuid.uuid4()))


@router.post("/ask", response_model=AIResponse)
def ask(req: QuestionRequest) -> AIResponse:
    try:
        logger.info("\nRecieved ask request: %s", req)
        chat_output = chat_service.chat(user_input=req.user_input, session_id=req.session_id)

        return chat_output
    except Exception as e:
        logger.error("\nAn error occured in chat interaction: %s", e)
        raise e
