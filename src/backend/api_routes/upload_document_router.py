"""This module defines the upload document API routes."""

import logging
from fastapi import APIRouter
from src.backend.api_routes.models.models import UserRequest, ChatResponse
from src.backend.services.chat_service.chat_service import ChatService

logger = logging.getLogger(__name__)

upload_document_router = APIRouter(prefix="/upload", tags=["upload"])


@upload_document_router.post("", response_model=ChatResponse)
async def upload_document(request) -> str:
    """Endpoint to handle document uploads."""
    raise NotImplementedError
