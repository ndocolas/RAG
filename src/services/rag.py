from src.services.vector_indexer import VectorIndexer


class RAGPipeline:
    """
    Simplified RAG pipeline:
      - Does NOT call an LLM
      - Returns only the most similar chunk (plain text)
    """

    def __init__(self) -> None:
        self.indexer = VectorIndexer()
        self.indexer.ensure_collection()

    def get_chunk(self, session_id: str, question: str) -> str:
        """
        Search for the top-1 chunk and return its text.
        """
        q_vec = self.indexer.embed_one(question)
        results = self.indexer.search(q_vec, top_k=1, session_id=session_id)

        if not results:
            return "No indexed content found for this chat."

        payload = (results[0].payload or {})
        chunk_text = (payload.get("text") or "").strip()
        return chunk_text or "No indexed content found for this chat."
