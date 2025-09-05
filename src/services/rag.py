from src.adapters.llm_gemini import GeminiClient
from src.services.embeddings import Embeddings
from src.services.vector_store import QdrantStore
from src.models.dto import Citation
from rag_prompt import SYSTEM_PROMPT
import time


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        blocks.append(f"[p.{c['page']}] {c['text']}")
    ctx = "\n\n".join(blocks)
    return f"{ctx}\n\nUser: {question}\nAnswer using ONLY the context above."


class RAGPipeline:
    def __init__(self):
        self.llm = GeminiClient()
        self.emb = Embeddings()
        self.store = QdrantStore()

    def ask(self, session_id: str, question: str, top_k: int = 5, mmr_lambda: float = 0.5, stream: bool = False):
        t0 = time.time()
        q_vec = self.emb.embed_batch([question])[0]
        # primeiro buscamos mais candidatos (8x) e aplicamos MMR localmente
        raw = self.store.search(q_vec, top_k*8, session_id)
        cand = [(i, r.vector) for i, r in enumerate(raw)]
        order = __import__("app.services.retriever").services.retriever.mmr(q_vec, cand, mmr_lambda, top_k)  # evita ciclo de import
        selected = [raw[i] for i in order]

        contexts = []
        citations = []
        for r in selected:
            payload = r.payload or {}
            contexts.append({
                "text": payload.get("text", ""),
                "page": payload.get("page", 1),
            })
            citations.append(Citation(
                source_id=payload.get("source_id", ""),
                page=payload.get("page", 1),
                score=float(r.score or 0.0),
                snippet=(payload.get("text", "")[:240] + "…") if payload.get("text") else ""
            ))

        prompt = build_prompt(question, contexts)
        if stream:
            return self.llm.stream(prompt, system=SYSTEM_PROMPT), citations, time.time()-t0
        else:
            out = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            return out, citations, time.time()-t0
