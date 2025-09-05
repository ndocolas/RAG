from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import os
import hashlib
from src.core.config import settings
from src.models.dto import UploadResponse
from src.services.pdf_parser import extract_pages
from src.services.chunker import make_chunks
from src.services.embeddings import Embeddings
from src.services.vector_store import QdrantStore

router = APIRouter()

UPLOAD_DIR = "/app/data/uploads"

@router.post("/documents", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    total_chunks = 0
    total_points = 0
    sources = []
    emb = Embeddings()
    store = QdrantStore()
    dim = settings.EMBEDDING_DIM
    store.ensure(dim)

    for f in files:
        ext = (f.filename.rsplit(".",1)[-1] or "").lower()
        if ext not in settings.ALLOWED_EXTS:
            raise HTTPException(400, f"Extensão não suportada: {ext}")
        if f.size and f.size > settings.MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"Arquivo excede {settings.MAX_FILE_MB} MB")

        path = os.path.join(UPLOAD_DIR, f.filename)
        with open(path, "wb") as out:
            out.write(await f.read())

        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        source_id = f"{f.filename}::{h}"
        pages = extract_pages(path)
        chunks = make_chunks(pages)
        texts = [c["text"] for c in chunks]
        vecs = emb.embed_batch(texts)

        points = []
        for v, c in zip(vecs, chunks):
            points.append({
                "vector": v,
                "payload": {
                    "session_id": session_id,
                    "source_id": source_id,
                    "page": c["page"],
                    "text": c["text"],
                }
            })
        store.upsert_points(points)
        total_chunks += len(chunks)
        total_points += len(points)
        sources.append(source_id)

    return UploadResponse(
        message="Indexação concluída.",
        session_id=session_id,
        docs_indexed=len(files),
        chunks=total_chunks,
        points=total_points,
        sources=sources
    )
