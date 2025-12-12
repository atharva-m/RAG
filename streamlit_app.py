import asyncio
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

# --- 1. Session & Identity Management (New) ---
if "user_session_id" not in st.session_state:
    # Generate a unique ID for this browser tab/session
    st.session_state.user_session_id = str(uuid.uuid4())

# --- Helper: Load Image as Base64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- Custom CSS Injection ---
def add_custom_css():
    # Attempt to load local image, fallback to gradient
    bin_str = get_base64_of_bin_file("/home/ubuntu/RAG/assets/background_image.jpeg")
    
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

    st.markdown(f"""
        <style>
        /* 1. Apply Background */
        {bg_image_css}

        /* 2. Global Typography */
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

        /* 3. CONTAINER STYLING */
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

        /* 4. WIDGET STYLING */
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
        
        /* 5. FILE UPLOADER */
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

        /* 6. ANSWER BOX */
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
        
        /* Hide Branding */
        div[data-testid="stSpinner"] {{ border: none !important; background: transparent !important; }}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# --- Logic Layer ---

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

async def send_rag_ingest_event(pdf_path: Path, unique_source_id: str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": unique_source_id, # Pass the unique ID to the backend
            },
        )
    )

async def send_rag_query_event(question: str, top_k: int, unique_source_id: str) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "source_id": unique_source_id, # Pass ID to restrict search
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
        data = resp.json()
        return data.get("data", [])
    except Exception:
        return []

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
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
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- UI Layout ---

st.markdown("<h1>RAG Agent</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Secure Document Analysis • Auto-Deletion Enabled (10m)</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

# LEFT COLUMN: Knowledge Base
with col1:
    st.markdown("<h3>Knowledge Base</h3>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload Document", 
        type=["pdf"], 
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="Limit 5MB per file • PDF"
    )
    
    if uploaded is not None:
        st.info(f"{uploaded.name}")
        status_box = st.empty()
        
        if st.button("Ingest Document", use_container_width=True):
            status_box.info("Saving file...")
            path = save_uploaded_pdf(uploaded)
            time.sleep(0.3)
            
            # Generate Unique ID for this specific file upload
            # Combines Session UUID + Filename to ensure total isolation
            unique_source_id = f"{st.session_state.user_session_id}--{uploaded.name}"
            
            # Save this ID to session state so we know what to query later
            st.session_state.active_source_id = unique_source_id
            
            status_box.info("Generating embeddings...")
            asyncio.run(send_rag_ingest_event(path, unique_source_id))
            
            status_box.success("Document added! It will self-destruct in 10 minutes.")
            time.sleep(2)
            status_box.empty()
            
    # Show active session info (optional, for debugging or clarity)
    if "active_source_id" in st.session_state:
        st.caption("Secure Session Active")
    else:
        st.caption("No active document")

# RIGHT COLUMN: Chat Interface
with col2:
    with st.container(border=True):
        st.markdown("<h3>Ask Questions</h3>", unsafe_allow_html=True)
        
        with st.form("rag_query_form", border=False):
            question = st.text_input(
                "Your Question",
                placeholder="What would you like to know?",
                label_visibility="collapsed"
            )
            
            c_label, c_input = st.columns([1, 1])
            with c_label:
                st.markdown("<div style='padding-top: 15px; font-weight: 500;'>Retrieval Depth</div>", unsafe_allow_html=True)
            with c_input:
                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    label_visibility="collapsed"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Generate Answer", use_container_width=True)

            if submitted and question.strip():
                # Check if user has actually uploaded something for this session
                if "active_source_id" not in st.session_state:
                     st.error("Please upload and ingest a document first.")
                else:
                    with st.spinner("Searching secure knowledge base..."):
                        try:
                            # Pass the Active Source ID to filter results
                            active_id = st.session_state.active_source_id
                            
                            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k), active_id))
                            
                            output = wait_for_run_output(event_id)
                            answer = output.get("answer", "")
                            sources = output.get("sources", [])
                            
                            st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                            st.markdown("<div class='answer-label'>Answer</div>", unsafe_allow_html=True)
                            st.markdown(answer or "No answer could be generated.")
                            st.markdown("</div>", unsafe_allow_html=True)

                            if sources:
                                st.markdown("**Sources**")
                                for idx, source in enumerate(sources, 1):
                                    # Clean up the display name (remove UUID prefix for cleaner UI)
                                    display_name = source.split("--")[-1] if "--" in source else source
                                    st.markdown(f"<div class='source-item'>{idx}. {display_name}</div>", unsafe_allow_html=True)
                                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
