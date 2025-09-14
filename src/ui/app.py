# flake8: noqa

"""Interface to interact with the RAG PDF application (native Streamlit full-page drop)."""

import json
import uuid
import requests
import streamlit as st
from src.backend.api_routes.models.models import UserRequest, ChatResponse
from src.backend.secrets.settings import settings

# =========================
# Page setup
# =========================
st.set_page_config(page_title="RAG PDF", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem;}
      .upload-box {
          border: 2px dashed #aaa;
          border-radius: 12px;
          padding: 2rem;
          text-align: center;
          color: #555;
          font-size: 1.1rem;
          margin-bottom: 1.5rem;
          background-color: #fafafa;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================
def ensure_session() -> str:
    """Ensure a unique session ID for the user."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def upload_files(session_id: str, files: list) -> dict:
    """Uploads files to the backend for indexing."""
    payload = {"session_id": session_id}
    to_send = [("files", (f.name, f.read(), "application/octet-stream")) for f in files]
    resp = requests.post(f"{settings.API_BASE_URL}/documents", data=payload, files=to_send, timeout=240)
    resp.raise_for_status()
    return resp.json()


def ask(session_id: str, user_input: str) -> dict:
    """Send a question to the backend and get a response."""
    try:
        req = UserRequest(session_id=session_id, user_input=user_input)
        resp = requests.post(f"{settings.API_BASE_URL}/chat", json=json.loads(req.model_dump_json()), timeout=240)

        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Error contacting backend: {e}")
        return {"error": str(e)}


# =========================
# State
# =========================
session_id = ensure_session()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_index_result" not in st.session_state:
    st.session_state.last_index_result = None

# =========================
# Upload Area
# =========================
st.title("Chat with your documents")

uploaded = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)


if uploaded:
    with st.spinner("Indexing file(s)..."):
        try:
            result = upload_files(session_id, uploaded)
            st.session_state.last_index_result = result
            st.success("Indexing completed!")
            st.json(result)
        except requests.RequestException as e:
            st.error(f"Indexing failed: {e}")

# =========================
# Chat
# =========================
st.subheader("Ask about your files")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask something about your indexed files…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            backend_response = ask(session_id, question)

            response_model = ChatResponse.model_validate(backend_response)

            placeholder.empty()

            response = (
                f"{response_model.response_model.response}\n\nReference: {response_model.response_model.reference}"
            )

            st.write(response)

            st.session_state.history.append({"role": "assistant", "content": response})
        except requests.RequestException as e:
            placeholder.error(f"Error calling the model: {e}")
