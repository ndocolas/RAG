from google import genai
from google.genai import types
from src.core.config import settings
import logging

log = logging.getLogger(__name__)


class Embeddings:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        cfg = types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIM)
        r = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,  # type: ignore
            config=cfg
        )
        return [e.values for e in r.embeddings]  # type: ignore
