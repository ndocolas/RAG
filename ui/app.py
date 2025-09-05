import json
import requests
import streamlit as st

# =========================
# Configurações e Constantes
# =========================
st.set_page_config(page_title="RAG PDF", layout="wide")
API_BASE = "http://api:8000/v1"


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
    to_send = [("files", (f.name, f.read(), "application/octet-stream")) for f in files]
    resp = requests.post(f"{API_BASE}/documents", data=payload, files=to_send, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ask(session_id: str, user_input: str) -> dict:
    """
    Chama o endpoint /ask (não-stream).
    Retorna o JSON completo (AIResponse).
    """
    payload = {"session_id": session_id, "user_input": user_input}
    resp = requests.post(f"{API_BASE}/ask", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


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

    # Chamada ao backend (não-stream) e renderização da resposta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            result = ask(session_id, question)
            final_text = result.get("response", "")
            if not final_text:
                final_text = "_Sem resposta do modelo._"

            placeholder.markdown(final_text)

            # Salva no histórico
            st.session_state.history.append({"role": "assistant", "content": final_text})

        except requests.RequestException as e:
            placeholder.error(f"Erro ao consultar o modelo: {e}")
