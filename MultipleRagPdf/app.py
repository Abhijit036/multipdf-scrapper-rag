import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_processor import process_pdfs
from utils.web_scraper import scrape_urls
from utils.vector_store import VectorStore
from utils.gemini_chain import GeminiRAGChain
from utils.prompt_templates import SYSTEM_PROMPT

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Nexus · Gemini",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent: #7c6af7;
    --accent2: #f76a8a;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --border: rgba(124,106,247,0.18);
    --glow: 0 0 30px rgba(124,106,247,0.15);
}

/* ── Base ── */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Title bar ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.6rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
}
.rag-header .logo { font-size: 2.4rem; }
.rag-header h1 {
    margin: 0;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
    background: linear-gradient(90deg, #7c6af7, #f76a8a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.rag-header p { margin: 0; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); }

/* ── Chat container ── */
.chat-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem;
    min-height: 420px;
    max-height: 560px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

/* ── Chat bubbles ── */
.msg-user, .msg-bot {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
    animation: fadeUp 0.3s ease;
}
.msg-user { flex-direction: row-reverse; }
.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar-user { background: linear-gradient(135deg, #7c6af7, #f76a8a); }
.avatar-bot  { background: linear-gradient(135deg, #0f3460, #533483); border: 1px solid var(--border); }
.bubble {
    max-width: 75%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.55;
}
.bubble-user {
    background: linear-gradient(135deg, rgba(124,106,247,0.25), rgba(247,106,138,0.15));
    border: 1px solid rgba(124,106,247,0.3);
    border-top-right-radius: 4px;
}
.bubble-bot {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
}
.source-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 0.4rem;
    padding: 0.2rem 0.5rem;
    background: rgba(124,106,247,0.08);
    border-radius: 4px;
    display: inline-block;
}

/* ── Input area ── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #533483) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,106,247,0.4) !important;
}

/* ── Stat pills ── */
.stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.75rem 0; }
.stat-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--accent);
}

/* ── Section headers ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.5rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alert ── */
.stAlert {
    background: rgba(124,106,247,0.1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.8; }
    100% { transform: scale(1.5); opacity: 0; }
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4ade80; display: inline-block;
    box-shadow: 0 0 6px #4ade80; margin-right: 6px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "chain": None,
        "doc_count": 0,
        "chunk_count": 0,
        "sources": [],
        "ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <div class="logo">🧠</div>
  <div>
    <h1>RAG Nexus</h1>
    <p>Multi-PDF · Web Scrape · Gemini API · Streamlit</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com/",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown('<div class="section-label">📄 PDF Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">🌐 Web URLs to Scrape</div>', unsafe_allow_html=True)
    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com\nhttps://docs.python.org",
        height=110,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">🎛️ Retrieval Settings</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        chunk_size   = st.number_input("Chunk Size",   200, 2000, 800, 100)
    with col2:
        chunk_overlap = st.number_input("Overlap",     0,    500,  150,  50)
    top_k = st.slider("Top-K Chunks", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    build_btn = st.button("🚀 Build Knowledge Base", use_container_width=True)

    if st.session_state.ready:
        st.markdown('<div class="section-label">📊 Index Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <span class="stat-pill">📂 {st.session_state.doc_count} sources</span>
          <span class="stat-pill">🧩 {st.session_state.chunk_count} chunks</span>
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#6b6b80;margin-top:0.4rem;">
        {'<br>'.join(f'• {s}' for s in st.session_state.sources[:8])}
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Build knowledge base ──────────────────────────────────────────────────────
if build_btn:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
    if not uploaded_files and not urls:
        st.warning("Upload at least one PDF or enter a URL.")
        st.stop()

    with st.spinner("Processing sources…"):
        all_docs = []
        source_labels = []

        if uploaded_files:
            pdf_docs = process_pdfs(uploaded_files, chunk_size, chunk_overlap)
            all_docs.extend(pdf_docs)
            source_labels.extend([f.name for f in uploaded_files])

        if urls:
            web_docs = scrape_urls(urls, chunk_size, chunk_overlap)
            all_docs.extend(web_docs)
            source_labels.extend(urls)

        vs = VectorStore()
        vs.build(all_docs)
        chain = GeminiRAGChain(
            vector_store=vs,
            top_k=top_k,
            temperature=temperature,
            system_prompt=SYSTEM_PROMPT,
        )

        st.session_state.vector_store = vs
        st.session_state.chain        = chain
        st.session_state.doc_count    = len(source_labels)
        st.session_state.chunk_count  = len(all_docs)
        st.session_state.sources      = source_labels
        st.session_state.ready        = True

    st.success(f"✅ Knowledge base ready — {len(all_docs)} chunks indexed!")
    st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────
if not st.session_state.ready:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#6b6b80;">
      <div style="font-size:3rem;margin-bottom:1rem;">🗂️</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;color:#e8e8f0;margin-bottom:0.5rem;">
        No knowledge base yet
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.75rem;">
        Upload PDFs and/or paste URLs in the sidebar,<br>then click <strong>Build Knowledge Base</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat history
    chat_html = '<div class="chat-wrap" id="chat-box">'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-user">
              <div class="avatar avatar-user">👤</div>
              <div class="bubble bubble-user">{msg["content"]}</div>
            </div>"""
        else:
            src_tags = "".join(
                f'<span class="source-tag">📎 {s}</span> '
                for s in msg.get("sources", [])
            )
            chat_html += f"""
            <div class="msg-bot">
              <div class="avatar avatar-bot">🧠</div>
              <div>
                <div class="bubble bubble-bot">{msg["content"]}</div>
                {f'<div style="margin-top:0.35rem">{src_tags}</div>' if src_tags else ''}
              </div>
            </div>"""
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Scroll to bottom JS
    st.markdown("""
    <script>
    const box = document.getElementById('chat-box');
    if(box) box.scrollTop = box.scrollHeight;
    </script>""", unsafe_allow_html=True)

    # Input row
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_query = st.text_input(
            "Ask anything…",
            key="query_input",
            placeholder="What does the document say about…?",
            label_visibility="collapsed",
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Thinking…"):
            result = st.session_state.chain.run(user_query)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()