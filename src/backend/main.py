"""Main application entry point for the RAG PDF FastAPI service."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.secrets.settings import settings
from src.backend.api_routes.v1_routes import v1_router

logging.basicConfig(level=settings.LOG_LEVEL.upper())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(v1_router, prefix="/v1")


@app.get("/")
async def root():
    """Root endpoint for the AI Orchestration Engine service."""
    return {
        "message": "Hello World! This is the RAG PDF FastAPI service.",
    }
