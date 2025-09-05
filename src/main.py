import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.models.v1.settings import settings
from src.api_routes.v1 import chat, documents

logging.basicConfig(level=settings.LOG_LEVEL.upper())

app = FastAPI(title="RAG PDF")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


app.include_router(documents.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")
