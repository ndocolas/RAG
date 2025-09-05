# RAG-PDF (Gemini)

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system that allows you to chat with PDF or TXT documents.  
Files are processed, split into chunks, converted into embeddings, and stored in **Qdrant**. During the conversation, the system retrieves the most relevant passages and sends them as context to **Gemini 2.5 Pro**, which replies in the same language as the question.

![RAG System Architecture Diagram](readme/ui-image.png) <!-- you can create/add an image here -->

## Features
- Document upload and processing (PDF/TXT)
- Text extraction with PyMuPDF
- Chunking (~800 tokens, 10% overlap)
- Embeddings with Gemini
- Vector storage in Qdrant
- Retrieval by similarity + MMR
- **Streaming** responses in the UI
- Citations with page and source snippet
- Interactive UI built with **Streamlit**

## Requirements
- Python 3.12
- Docker + Docker Compose

## Dependencies
Main packages:
- `fastapi` – API server
- `uvicorn` – ASGI server
- `pydantic-settings` – Config via `.env`
- `google-genai` – Gemini client
- `qdrant-client` – Vector DB client
- `pymupdf` – PDF parsing
- `numpy` – Similarity + MMR
- `sse-starlette` – SSE streaming in FastAPI
- `streamlit` – Web UI
- `sseclient-py` – SSE consumption in the client

## Docker Deployment

1. Clone the repository
    ```bash
    git clone https://github.com/your-username/rag-pdf-gemini.git
    cd rag-pdf-gemini
    ```

2. Set up environment variables  
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_key
   GEMINI_MODEL=gemini-2.5-pro
   EMBEDDING_MODEL=gemini-embedding-001
   EMBEDDING_DIM=768
   QDRANT_URL=http://qdrant:6333
   QDRANT_COLLECTION=docs
   MAX_FILE_MB=15
   LOG_LEVEL=INFO
   ```

3. Start the containers
    ```bash
    docker compose down -v --remove-orphans
    docker compose up --build
    ```

4. Access the services:
    - **Web UI (Streamlit):** http://localhost:8501  
    - **API (FastAPI docs):** http://localhost:8000/docs  
    - **Qdrant Dashboard:** http://localhost:6333/dashboard  

## Services
The `docker-compose.yml` launches three services:
- **API server:** FastAPI backend for document processing and queries
- **Web UI:** Streamlit interface for chatting with PDFs
- **Vector Database:** Qdrant for storing embeddings

## API Endpoints

1. **Start Chat**
   - `POST /v1/start_chat`
   - Creates a new chat session
   - Returns a `session_id`

2. **Upload Documents**
   - `POST /v1/documents`
   - Upload and index PDFs/TXT for a session
   - Requires `session_id` and files
   - Returns indexing statistics

3. **Ask Question**
   - `POST /v1/ask`
   - Ask a question with a full response
   - Requires `session_id` and `question`

4. **Ask Question (Streaming)**
   - `POST /v1/ask/stream`
   - Returns SSE streaming response
   - Includes citations (page/snippet)

### Endpoint Details

| Endpoint          | Method | Content-Type       | Body/Params |
|-------------------|--------|--------------------|-------------|
| `/v1/start_chat`  | POST   | -                  | None        |
| `/v1/documents`   | POST   | multipart/form-data| `session_id`, `files` |
| `/v1/ask`         | POST   | application/json   | `{"question": "text", "session_id": "uuid"}` |
| `/v1/ask/stream`  | POST   | application/json   | `{"question": "text", "session_id": "uuid"}` |

## Project Structure
```
RAG-PDF-Gemini/
├── docker/
│   ├── Dockerfile.api       # API container definition
│   └── Dockerfile.ui        # UI container definition
├── src/
│   ├── adapters/            # External adapters (Gemini client)
│   ├── api_routes/          # FastAPI routes
│   ├── core/                # Core config and utils
│   ├── models/              # Pydantic DTOs
│   ├── services/            # Core logic: parser, chunker, embeddings, retriever, RAG
│   ├── utils/               # Helpers
│   └── main.py              # FastAPI entrypoint
├── ui/
│   ├── app.py               # Streamlit interface
│   └── requirements.txt     # UI dependencies
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # API dependencies
├── .env.example             # Env example
├── README.md                # This file
└── LICENSE
```

## Additional Resources
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Docs](https://docs.streamlit.io/)
