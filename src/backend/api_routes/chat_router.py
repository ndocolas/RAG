"""API routes for chat interactions."""

import logging
from fastapi import APIRouter
from src.backend.api_routes.models.models import UserRequest, ChatResponse
from src.backend.services.chat_service.chat_service import ChatService

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])

chat_service = ChatService()


@chat_router.post("", response_model=ChatResponse)
async def ask(request: UserRequest) -> ChatResponse:
    """
    Endpoint to handle user chat requests within a session.

    Inputs:
    request: UserRequest object containing the session_id and user_input

    Returns:
    ChatResponse: The structured response containing the session_id, user input, and the model-generated response
    """
    try:
        logger.info("\nReceived ask request: %s", request)
        response = await chat_service.chat(
            user_input=request.user_input,
            session_id=request.session_id
        )

        logger.debug("\nRequest retrieved: %s", response)

        return ChatResponse(
            session_id=response.session_id,
            user_input=response.user_input,
            response_model=response.response_model
        )
    except Exception as e:
        logger.error("\nAn error occurred in chat interaction: %s", e)
        raise e
