from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.core.config import settings
import uuid


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)

    def ensure(self, dim: int):
        # cria se não existir
        try:
            self.client.get_collection(settings.QDRANT_COLLECTION)
        except Exception:
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert_points(self, points: list[dict]):
        qpoints = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=p["vector"],
                payload=p["payload"],
            ) for p in points
        ]
        self.client.upsert(collection_name=settings.QDRANT_COLLECTION, points=qpoints)

    def search(self, vector: list[float], top_k: int, session_id: str):
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        return self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            query_filter=flt,
            with_vectors=True,
            with_payload=True
        )
