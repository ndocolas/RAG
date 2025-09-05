from typing import List, Dict
from src.models.v1.settings import settings
import fitz


def _approx_chars_for_tokens(tokens: int) -> int:
    """Rough heuristic: 1 token ≈ 4 characters."""
    return tokens * 4


def make_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Convert pages into chunks of approximate token size (using character proxy),
    with overlap configured in settings.
    """
    tgt = _approx_chars_for_tokens(settings.CHUNK_TARGET_TOKENS)
    overlap = int(tgt * settings.CHUNK_OVERLAP)

    chunks: List[Dict] = []
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


# ---------- Parsing ----------
def extract_pages(path: str) -> List[Dict]:
    """
    Extract text per page from a PDF (using PyMuPDF) or load a TXT file
    as a single "page".
    """
    out: List[Dict] = []
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            out.append({"page": 1, "text": text})
        return out

    doc = fitz.open(path)
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            text = page.get_text("text").strip()  # type: ignore
            if text:
                out.append({"page": pno + 1, "text": text})
    finally:
        doc.close()

    return out
