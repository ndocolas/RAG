import json
import requests
import streamlit as st
import sseclient


# =========================
# Configurações e Constantes
# =========================
st.set_page_config(page_title="RAG PDF (Gemini)", layout="wide")
API_BASE = st.secrets.get("API_URL", "http://api:8000/v1")


# =========================
# Funções Auxiliares
# =========================
def ensure_session() -> str:
    """Garante que exista um session_id no estado da UI."""
    if "session_id" not in st.session_state:
        resp = requests.post(f"{API_BASE}/start_chat", timeout=15)
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]
    return st.session_state.session_id


def upload_files(session_id: str, files: list) -> dict:
    """Envia arquivos para indexação no backend."""
    payload = {"session_id": session_id}
    # requests quer (name, fileobj, mime); usamos "application/octet-stream" por segurança
    to_send = [("files", (f.name, f.read(), "application/octet-stream")) for f in files]
    resp = requests.post(f"{API_BASE}/documents", data=payload, files=to_send, timeout=120)
    resp.raise_for_status()
    return resp.json()


def stream_answer(session_id: str, question: str):
    """
    Faz a pergunta ao endpoint de streaming e gera tokens conforme chegam.
    Retorna (texto_final, citations_json).
    """
    payload = {"session_id": session_id, "question": question}
    resp = requests.post(f"{API_BASE}/ask/stream", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    client = sseclient.SSEClient(resp)  # type: ignore
    full_text = ""
    citations = None

    for evt in client.events():
        if evt.event == "citations":
            citations = json.loads(evt.data) if evt.data else []
        elif evt.event == "token":
            full_text += evt.data or ""
            yield ("token", full_text)  # atualiza a UI a cada pedaço
        elif evt.event == "done":
            break

    yield ("done", (full_text, citations))


def render_citations(citations: list[dict]):
    """Renderiza a seção de citações."""
    if not citations:
        return
    with st.expander("Citações"):
        for c in citations:
            page = c.get("page", 1)
            score = c.get("score", 0.0)
            src = c.get("source_id", "")
            st.markdown(f"- **p.{page}** — _score {score:.3f}_ — `{src}`")


# =========================
# Estado Inicial
# =========================
session_id = ensure_session()
if "history" not in st.session_state:
    # Armazena turnos simples: [{"role":"user"/"assistant","content":str}]
    st.session_state.history = []


# =========================
# Barra Lateral (Upload)
# =========================
st.sidebar.header("Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader(
    "PDF/TXT até 15 MB",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button("Indexar", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("Selecione ao menos um arquivo.")
    else:
        try:
            result = upload_files(session_id, uploaded_files)
            st.sidebar.success("Indexação concluída!")
            st.sidebar.json(result)
        except requests.RequestException as e:
            st.sidebar.error(f"Falha ao indexar: {e}")


# =========================
# Chat
# =========================
st.title("Chat sobre o PDF")

# Histórico visual
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrada do usuário
question = st.chat_input("Pergunte algo sobre seus arquivos...")

if question:
    # Mostra pergunta no histórico
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Área de resposta com streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            final_text, final_citations = "", None
            # Consome o stream e atualiza o placeholder
            for kind, payload in stream_answer(session_id, question):
                if kind == "token":
                    placeholder.markdown(payload)
                elif kind == "done":
                    final_text, final_citations = payload

            # Congela o texto final e renderiza citações
            placeholder.markdown(final_text)
            render_citations(final_citations)  # type: ignore

            # Salva no histórico
            st.session_state.history.append({"role": "assistant", "content": final_text})

        except requests.RequestException as e:
            placeholder.error(f"Erro ao consultar o modelo: {e}")
