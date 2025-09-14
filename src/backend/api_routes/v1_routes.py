"""This module sets up the API v1 routes of the AI Orchestration Engine."""

from fastapi import APIRouter

from src.backend.api_routes.chat_router import chat_router
from src.backend.api_routes.upload_document_router import upload_document_router

v1_router = APIRouter()
v1_router.include_router(chat_router)
v1_router.include_router(upload_document_router)
