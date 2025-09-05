import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api_routes import documents, chat
from src.core.config import settings

logging.basicConfig(level=settings.LOG_LEVEL.upper())

app = FastAPI(title="RAG PDF (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(documents.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")
