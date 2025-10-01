"""This module defines the upload document API routes."""

import logging
from typing import List, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from src.backend.services.vector_service.vector_service import RetrievalService

logger = logging.getLogger(__name__)

upload_document_router = APIRouter(tags=["documents"])

retrieval = RetrievalService()


@upload_document_router.post("/documents")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
) -> dict:
    """Endpoint to receive uploaded files, index them with FAISS, and persist them per session.

    Inputs:
    session_id: Unique session identifier to associate with the indexed documents
    files: List of uploaded files (PDF or TXT) provided via multipart form-data

    Returns:
    dict: Summary of indexing results containing the session_id, indexed file names, and total chunk count
    """
    try:
        logger.info("Uploading %d file(s) for session_id=%s", len(files), session_id)

        # read bytes from each UploadFile
        file_tuples: List[Tuple[str, bytes]] = []
        for f in files:
            content = await f.read()
            file_tuples.append((f.filename, content))  # type: ignore

        result = retrieval.upsert_files(session_id, file_tuples)
        logger.info("Indexing done: %s", result)
        return result
    except Exception as e:
        logger.exception("Failed to upload/index documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
