import logging
import uuid

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from google import genai
from google.genai import types
from src.models.v1.settings import settings
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

log = logging.getLogger(__name__)


class VectorIndexer:
    """
    Unified service for:
      - Generating embeddings with Gemini
      - Indexing and searching vectors in Qdrant
    """

    def __init__(self) -> None:
        self._gclient = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._embed_model = settings.EMBEDDING_MODEL
        self._embed_dim = settings.EMBEDDING_DIM

        # Qdrant
        self._qclient = QdrantClient(url=settings.QDRANT_URL)
        self._collection = settings.QDRANT_COLLECTION

    # ---------- Qdrant ----------
    def ensure_collection(self) -> None:
        """Create the collection if it does not exist. Idempotent."""
        try:
            self._qclient.get_collection(self._collection)
            return
        except Exception:
            pass

        self._qclient.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._embed_dim, distance=Distance.COSINE),
        )
        log.info("Qdrant collection %s ensured (dim=%d).", self._collection, self._embed_dim)

    def upsert_chunks(
        self,
        session_id: str,
        chunks: List[Dict],
        source_id: Optional[str] = None
    ) -> int:
        """
        Take chunks (each with 'text' and 'page'), generate embeddings in batch,
        and insert into Qdrant with payload containing session_id, page, text, and source_id.
        Returns the number of points inserted.
        """
        texts = [c["text"] for c in chunks]
        vecs = self.embed_batch(texts)

        points = []
        for v, c in zip(vecs, chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v,
                    payload={
                        "session_id": session_id,
                        "page": c.get("page", 1),
                        "text": c.get("text", ""),
                        "source_id": source_id or "",
                    },
                )
            )

        self._qclient.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query_vec: List[float], top_k: int, session_id: str):
        """
        Vector search restricted by session_id.
        Returns client objects with .payload/.vector/.score.
        """
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        return self._qclient.search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=flt,
            with_vectors=True,
            with_payload=True,
        )

    # ---------- Embeddings ----------
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cfg = types.EmbedContentConfig(output_dimensionality=self._embed_dim)
        try:
            resp = self._gclient.models.embed_content(
                model=self._embed_model,
                contents=texts,
                config=cfg
            )
            return [e.values for e in resp.embeddings]  # type: ignore
        except Exception:
            # fallback: um por vez (mais lento, mas robusto)
            vecs: List[List[float]] = []
            for t in texts:
                single = self._gclient.models.embed_content(
                    model=self._embed_model,
                    contents=t,
                       config=cfg
                )
                vecs.append(single.embedding.values)  # type: ignore
            return vecs


    def embed_one(self, text: str) -> List[float]:
        """Generate embeddings for a single string."""
        return self.embed_batch([text])[0]
