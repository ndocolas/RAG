from typing import List, Dict
import fitz


def extract_pages(path: str) -> List[Dict]:
    out, order = [], 0
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            out.append({"page": 1, "text": text, "order": order})
        return out

    doc = fitz.open(path)
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            text = page.get_text("text").strip()  # type: ignore
            if text:
                out.append({"page": pno+1, "text": text, "order": order})
                order += 1
    finally:
        doc.close()
    return out
