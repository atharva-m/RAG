import asyncio
import json
from pathlib import Path
import time
import os
import base64
import requests
import streamlit as st
import inngest
import uuid
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
st.set_page_config(page_title="RAG Agent", layout="wide", initial_sidebar_state="collapsed")

# How long documents live in Qdrant (must match the `sleep` in main.py).
DOC_TTL_SECONDS = 600  # 10 minutes

# File that persists active-document state across page refreshes.
# In production, replace with Redis or a proper DB.
SESSION_STORE = Path("session_store.json")


# ---------------------------------------------------------------------------
# Session Persistence
# The session ID lives in the URL (?sid=...) so it survives page refreshes.
# Active document metadata is written to SESSION_STORE on every change so it
# can be restored when the same URL is reopened.
# ---------------------------------------------------------------------------

def _load_all_sessions() -> dict:
    if SESSION_STORE.exists():
        try:
            return json.loads(SESSION_STORE.read_text())
        except Exception:
            return {}
    return {}


def _write_all_sessions(sessions: dict):
    SESSION_STORE.write_text(json.dumps(sessions))


def _load_session_docs(session_id: str) -> dict:
    """
    Load active docs for this session, automatically pruning any that have
    exceeded DOC_TTL_SECONDS (i.e. whose Qdrant vectors have already been wiped).

    Returns: {source_id: {"filename": str, "ingested_at": float}}
    """
    all_sessions = _load_all_sessions()
    docs = all_sessions.get(session_id, {})
    now = time.time()
    active = {
        k: v for k, v in docs.items()
        if now - v.get("ingested_at", 0) < DOC_TTL_SECONDS
    }
    if len(active) != len(docs):
        all_sessions[session_id] = active
        _write_all_sessions(all_sessions)
    return active


def _persist_session_docs(session_id: str, docs: dict):
    all_sessions = _load_all_sessions()
    all_sessions[session_id] = docs
    _write_all_sessions(all_sessions)


def get_or_create_session_id() -> str:
    """
    Read the session ID from the URL query param `sid`.
    If absent (first visit), generate a new UUID and write it to the URL so
    subsequent refreshes land on the same session.
    """
    params = st.query_params
    if "sid" in params:
        return params["sid"]
    sid = str(uuid.uuid4())
    st.query_params["sid"] = sid
    return sid


# Boot: establish session identity before any UI code runs.
session_id = get_or_create_session_id()

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = session_id

# Restore active docs from disk (prunes expired ones automatically).
if "active_docs" not in st.session_state:
    st.session_state.active_docs = _load_session_docs(session_id)
    # active_docs shape: {source_id: {"filename": str, "ingested_at": float}}


# ---------------------------------------------------------------------------
# Helper: load background image as base64
# ---------------------------------------------------------------------------

def get_base64_of_bin_file(bin_file: str) -> str | None:
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def add_custom_css():
    # Prefer an env-configured path; fall back to gradient if not found.
    bg_path = os.getenv("BACKGROUND_IMAGE_PATH", "assets/background_image.jpeg")
    bin_str = get_base64_of_bin_file(bg_path)

    if bin_str:
        bg_image_css = f"""
            .stApp {{
                background-image: url("data:image/jpeg;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        """
    else:
        bg_image_css = """
            .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        """

    st.markdown(
        f"""
        <style>
        {bg_image_css}

        .stApp, p, h1, h2, h3, label, .stMarkdown {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #2c3e50 !important;
        }}

        h1 {{
            font-weight: 700;
            letter-spacing: 2px;
            font-size: 3.5rem !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem !important;
            text-align: center;
        }}

        h3 {{
            color: #667eea !important;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }}

        .subtitle {{
            text-align: center;
            color: #5a6c7d !important;
            font-size: 1.1rem;
            margin-bottom: 3rem;
            font-weight: 300;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255, 255, 255, 0.35);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}

        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.6);
            border: 2px solid rgba(255,255,255,0.8);
            color: #2c3e50;
            border-radius: 12px;
        }}
        .stNumberInput > div > div > input {{
            background-color: #ffffff !important;
            border: 2px solid #ccc !important;
            color: #000000 !important;
            border-radius: 8px;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            width: 100%;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
        }}
        .stButton > button p {{
            color: #ffffff !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}

        [data-testid="stFileUploader"] {{
            background-color: transparent !important;
        }}
        [data-testid="stFileUploader"] section {{
            background-color: transparent !important;
            border: 2px dashed #000 !important;
        }}
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] div {{
            color: #000000 !important;
        }}
        [data-testid="stFileUploader"] button {{
            color: #ffffff !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }}

        .answer-box {{
            background: rgba(255, 255, 255, 0.85);
            border-left: 5px solid #667eea;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            animation: fadeIn 0.5s ease-in;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .answer-label {{
            font-weight: 700;
            color: #667eea !important;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }}

        .doc-item {{
            background: rgba(255,255,255,0.6);
            border-radius: 8px;
            padding: 6px 10px;
            margin-bottom: 6px;
            font-size: 0.9rem;
            border-left: 3px solid #667eea;
        }}

        div[data-testid="stSpinner"] {{ border: none !important; background: transparent !important; }}
        header {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


add_custom_css()


# ---------------------------------------------------------------------------
# Logic Layer
# ---------------------------------------------------------------------------

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file, sid: str) -> Path:
    """
    Save to a session-scoped subdirectory so two users uploading a file
    with the same name never collide on disk.
    """
    uploads_dir = Path("uploads") / sid
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path, source_id: str, filename: str) -> str:
    """Fire the ingest event and return the Inngest event ID for polling."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": source_id,
                "filename": filename,
            },
        )
    )
    return result[0]  # event ID


async def send_rag_query_event(
    question: str, top_k: int, source_ids: list[str]
) -> str:
    """Fire the query event and return the Inngest event ID for polling."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "source_ids": source_ids,   # list — server does OR filter
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def wait_for_run_output(
    event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5
) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )
        time.sleep(poll_interval_s)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.markdown("<h1>RAG Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Secure Document Analysis • Auto-Deletion Enabled (10m)</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: Knowledge Base ---
with col1:
    st.markdown("<h3>Knowledge Base</h3>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Document",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="Limit 5MB per file • PDF",
    )

    if uploaded is not None:
        st.info(f"{uploaded.name}")
        status_box = st.empty()

        if st.button("Ingest Document", use_container_width=True):
            # Build a source_id that is globally unique regardless of filename.
            # We use the session ID as a namespace and a UUID so two uploads of
            # the same filename in the same session get distinct source_ids.
            doc_uuid = str(uuid.uuid4())
            source_id = f"{session_id}::{doc_uuid}"
            filename = uploaded.name

            status_box.info("Saving file…")
            # Session-scoped path prevents filename collisions between users.
            path = save_uploaded_pdf(uploaded, session_id)

            status_box.info("Generating embeddings…")
            event_id = asyncio.run(
                send_rag_ingest_event(path, source_id, filename)
            )

            # Poll until the ingest job completes (vectors are in Qdrant)
            # before showing success. Without this the user can query before
            # the document is actually indexed.
            status_box.info("Indexing document…")
            try:
                wait_for_run_output(event_id, timeout_s=120.0)
                # Persist the new doc in session state and on disk.
                st.session_state.active_docs[source_id] = {
                    "filename": filename,
                    "ingested_at": time.time(),
                }
                _persist_session_docs(session_id, st.session_state.active_docs)
                status_box.success(
                    f"'{filename}' indexed! Auto-deletes in 10 minutes."
                )
            except Exception as e:
                status_box.error(f"Ingestion failed: {e}")

            time.sleep(2)
            status_box.empty()

    # Show active documents for this session.
    active_docs: dict = st.session_state.active_docs
    if active_docs:
        st.caption(f"Active documents ({len(active_docs)})")
        now = time.time()
        for sid, meta in active_docs.items():
            elapsed = now - meta.get("ingested_at", now)
            remaining = max(0, int(DOC_TTL_SECONDS - elapsed))
            mins, secs = divmod(remaining, 60)
            st.markdown(
                f"<div class='doc-item'>📄 {meta['filename']}"
                f"<br><small>⏱ {mins}m {secs:02d}s remaining</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No active documents")


# --- RIGHT COLUMN: Chat Interface ---
with col2:
    with st.container(border=True):
        st.markdown("<h3>Ask Questions</h3>", unsafe_allow_html=True)

        with st.form("rag_query_form", border=False):
            question = st.text_input(
                "Your Question",
                placeholder="What would you like to know?",
                label_visibility="collapsed",
            )

            c_label, c_input = st.columns([1, 1])
            with c_label:
                st.markdown(
                    "<div style='padding-top: 15px; font-weight: 500;'>Retrieval Depth</div>",
                    unsafe_allow_html=True,
                )
            with c_input:
                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    label_visibility="collapsed",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Generate Answer", use_container_width=True
            )

            if submitted and question.strip():
                active_docs = st.session_state.active_docs
                if not active_docs:
                    st.error("Please upload and ingest a document first.")
                else:
                    with st.spinner("Searching secure knowledge base…"):
                        try:
                            # Pass ALL source IDs for this session so the
                            # backend performs an OR-filter across every
                            # document the user has uploaded.
                            source_ids = list(active_docs.keys())

                            event_id = asyncio.run(
                                send_rag_query_event(
                                    question.strip(), int(top_k), source_ids
                                )
                            )
                            output = wait_for_run_output(event_id)

                            answer = output.get("answer", "")
                            sources = output.get("sources", [])

                            st.markdown(
                                "<div class='answer-box'>", unsafe_allow_html=True
                            )
                            st.markdown(
                                "<div class='answer-label'>Answer</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer or "No answer could be generated.")
                            st.markdown("</div>", unsafe_allow_html=True)

                            if sources:
                                st.markdown("**Sources**")
                                for idx, source in enumerate(sources, 1):
                                    # Sources are already clean filenames —
                                    # stored in the payload at ingest time,
                                    # no string-parsing required.
                                    st.markdown(
                                        f"<div class='doc-item'>{idx}. {source}</div>",
                                        unsafe_allow_html=True,
                                    )

                        except Exception as e:
                            st.error(f"Error: {e}")
