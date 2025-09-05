import os
import hashlib
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.models.v1.settings import settings
from src.models.v1.dto import UploadResponse
from src.services.ingest import extract_pages, make_chunks
from src.services.vector_indexer import VectorIndexer

router = APIRouter()
logger = logging.getLogger(__name__)

indexer = VectorIndexer()
UPLOAD_DIR = "/app/data/uploads"

@router.post("/documents", response_model=UploadResponse)
async def upload_and_index_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    logger.info("\nReceived /documents request: session_id=%s", session_id)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    indexer.ensure_collection()

    total_chunks = 0
    total_points = 0
    sources: list[str] = []

    for f in files:
        ext = (f.filename.rsplit(".", 1)[-1] or "").lower()  # type: ignore
        if ext not in settings.ALLOWED_EXTS:
            raise HTTPException(400, f"Unsupported extension: {ext}")

        # size guard (if provided by client)
        max_bytes = settings.MAX_FILE_MB * 1024 * 1024
        if hasattr(f, "size") and getattr(f, "size", None) and f.size > max_bytes:  # type: ignore
            raise HTTPException(413, f"File exceeds {settings.MAX_FILE_MB} MB")

        # Save & hash in one pass
        path = os.path.join(UPLOAD_DIR, f.filename)  # type: ignore
        hasher = hashlib.sha256()
        with open(path, "wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)

        source_id = f"{f.filename}::{hasher.hexdigest()[:16]}"

        pages = extract_pages(path)
        chunks = make_chunks(pages)
        inserted = indexer.upsert_chunks(session_id=session_id, chunks=chunks, source_id=source_id)

        total_chunks += len(chunks)
        total_points += inserted
        sources.append(source_id)

    resp = UploadResponse(
        message="Indexing completed.",
        session_id=session_id,
        docs_indexed=len(files),
        chunks=total_chunks,
        points=total_points,
        sources=sources,
    )
    logger.info("\nDocuments ingested successfully: %s", resp)
    return resp
