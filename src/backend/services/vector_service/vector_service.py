from __future__ import annotations
import io
import os
from typing import List, Tuple

from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.backend.secrets.settings import settings


INDEX_ROOT = os.path.join(os.getcwd(), "faiss_indexes")
EMBED_MODEL = settings.EMBEDDING_MODEL


class RetrievalService(BaseModel):
    """Service to handle vector operations per session (build/load/search)."""

    def __init__(self):
        super().__init__()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )
        os.makedirs(INDEX_ROOT, exist_ok=True)

    def upsert_files(self, session_id: str, files: List[Tuple[str, bytes]]) -> dict:
        """
        Ingest a list of (filename, bytes), extract text, split into chunks, index with FAISS, and persist per session.

        Inputs:
        session_id: Unique session identifier used to locate the FAISS index directory
        files: List of tuples (filename, file_bytes) to be parsed and indexed. Supports PDF and TXT.

        Returns:
        dict: A summary with the session_id, the set of files indexed, and the total number of chunks stored
        """
        texts, metadatas = self._extract_texts_with_meta(files)
        chunk_texts, chunk_metas = self._split_with_meta(texts, metadatas)

        vs = self._load_index(session_id)
        if vs is None:
            vs = FAISS.from_texts(
                chunk_texts,
                embedding=self.embeddings,
                metadatas=chunk_metas,
            )
        else:
            vs.add_texts(chunk_texts, metadatas=chunk_metas)

        self._save_index(session_id, vs)

        return {
            "session_id": session_id,
            "files_indexed": list({m["source"] for m in chunk_metas}),
            "chunks_count": len(chunk_texts),
        }

    def top_context(self, session_id: str, query: str, k: int = 1) -> Tuple[str, List[Document]]:
        """
        Run a similarity search against the session's index and return concatenated context and raw docs.

        Inputs:
        session_id: Unique session identifier linked to the persisted FAISS index
        query: Natural-language query to search similar chunks
        k: Number of top documents to retrieve (defaults to 1)

        Returns:
        Tuple[str, List[Document]]: The formatted context string and the list of retrieved LangChain documents
        """
        vs = self._load_index(session_id)
        if vs is None:
            raise RuntimeError("No index for this session. Upload documents first.")

        docs = vs.similarity_search(query, k=k)
        context = self._format_context(docs)
        return context, docs

    def _index_dir(self, session_id: str) -> str:
        """
        Build the absolute path to the FAISS index directory for a given session.

        Inputs:
        session_id: Unique session identifier

        Returns:
        str: Absolute path to the session-specific index folder
        """
        return os.path.join(INDEX_ROOT, session_id)

    def _load_index(self, session_id: str):
        """
        Load a FAISS vector store from disk for the given session if it exists.

        Inputs:
        session_id: Unique session identifier whose index should be loaded

        Returns:
        FAISS | None: The loaded vector store or None if no index directory exists
        """
        index_dir = self._index_dir(session_id)
        if not os.path.exists(index_dir):
            return None
        # In local environments, FAISS requires allow_dangerous_deserialization=True to load a saved index
        return FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _save_index(self, session_id: str, vs: FAISS) -> None:
        """
        Persist a FAISS vector store to the session-specific directory.

        Inputs:
        session_id: Unique session identifier whose index directory will be used
        vs: FAISS vector store to save

        Returns:
        None: This function writes files to disk and does not return a value
        """
        vs.save_local(self._index_dir(session_id))

    def _extract_texts_with_meta(self, files: List[Tuple[str, bytes]]) -> Tuple[List[str], List[dict]]:
        """
        Extract text and metadata from uploaded files into aligned lists.

        Inputs:
        files: List of tuples (filename, file_bytes). Supports PDF (page-level) and TXT (file-level).

        Returns:
        Tuple[List[str], List[dict]]: Parallel lists where texts[i] aligns with metadatas[i]; metadata includes 'source' and 'page'
        """
        texts, metas = [], []
        for filename, data in files:
            lower = filename.lower()
            if lower.endswith(".pdf"):
                pdf_reader = PdfReader(io.BytesIO(data))
                for i, page in enumerate(pdf_reader.pages):
                    content = page.extract_text() or ""
                    if content.strip():
                        texts.append(content)
                        metas.append({"source": filename, "page": i + 1})
            elif lower.endswith(".txt"):
                text = (data.decode("utf-8", errors="ignore")).strip()
                if text:
                    texts.append(text)
                    metas.append({"source": filename, "page": None})
        return texts, metas

    def _split_with_meta(
        self, texts: List[str], metas: List[dict], chunk_size: int = 1200, chunk_overlap: int = 150
    ) -> Tuple[List[str], List[dict]]:
        """
        Split texts into chunks while propagating and extending metadata.

        Inputs:
        texts: List of raw text strings to split
        metas: List of metadata dicts aligned with 'texts'
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap size between adjacent chunks

        Returns:
        Tuple[List[str], List[dict]]: Chunked texts and corresponding metadata with an added 'chunk_id'
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        out_texts, out_metas = [], []
        for text, meta in zip(texts, metas):
            chunks = splitter.split_text(text)
            for idx, ch in enumerate(chunks):
                m = dict(meta)
                m["chunk_id"] = idx
                out_texts.append(ch)
                out_metas.append(m)
        return out_texts, out_metas

    def _format_context(self, docs: List[Document]) -> str:
        """
        Concatenate retrieved documents into a single context string with source references.

        Inputs:
        docs: List of retrieved LangChain Document objects

        Returns:
        str: A context string containing the content of each document preceded by its source/page tag
        """
        blocks = []
        for d in docs:
            src = d.metadata.get("source")
            pg = d.metadata.get("page")
            tag = f"{src} (p.{pg})" if (src and pg) else (src or "—")
            blocks.append(f"[{tag}]\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)
