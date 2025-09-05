from typing import List, Dict
from src.core.config import settings


def approx_chars_for_tokens(tokens: int) -> int:
    return tokens * 4


def make_chunks(pages: List[Dict]) -> List[Dict]:
    tgt = approx_chars_for_tokens(settings.CHUNK_TARGET_TOKENS)
    overlap = int(tgt * settings.CHUNK_OVERLAP)
    chunks = []
    for p in pages:
        text, page = p["text"], p["page"]
        start = 0
        while start < len(text):
            end = min(start + tgt, len(text))
            piece = text[start:end]
            if piece.strip():
                chunks.append({
                    "text": piece.strip(),
                    "page": page,
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks
