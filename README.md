## 🌐 Live Demo

👉 https://multipdf-scrapper-chatbot.streamlit.app/

> Try the app: Upload PDFs or paste URLs and start chatting instantly 🚀

# 🧠 RAG Nexus — Multi-PDF + Web Scrape RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot powered by
**Google Gemini**, supporting **PDF uploads** and **live web scraping**, with
advanced **prompt engineering**, deployable to **Streamlit Cloud** in minutes.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-PDF ingestion** | Upload multiple PDFs simultaneously |
| **Web scraping** | Paste any public URLs; HTML is cleaned and indexed |
| **TF-IDF retrieval** | Fast in-memory cosine-similarity search |
| **Gemini 2.0 Flash** | Fast, accurate generation (swap to 1.5-pro anytime) |
| **Prompt engineering** | System persona · Few-shot · Chain-of-thought · Citation forcing · Hallucination guard |
| **Streamlit UI** | Dark-themed, responsive chat interface |
| **Configurable** | Chunk size, overlap, top-k, temperature all adjustable |

---

## 🗂️ Project Structure

```
rag_chatbot/
├── app.py                   # Streamlit entry point
├── requirements.txt
├── .env.example             # Copy to .env for local dev
├── .streamlit/
│   └── config.toml          # Theme & server settings
└── utils/
    ├── __init__.py
    ├── pdf_processor.py     # PDF text extraction & chunking
    ├── web_scraper.py       # HTML fetching & cleaning
    ├── vector_store.py      # TF-IDF vector index
    ├── gemini_chain.py      # RAG orchestration (retrieve → prompt → generate)
    └── prompt_templates.py  # All prompt engineering templates
```

---

## 🚀 Local Setup

### 1. Clone / download
```bash
git clone <your-repo-url>
cd rag_chatbot
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# GEMINI_API_KEY=AIza...
```
Get a free key at https://aistudio.google.com/

### 5. Run
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a **GitHub repository**.
2. Go to https://share.streamlit.io → **New app** → select your repo.
3. Set **Main file path** to `app.py`.
4. In **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy** — done!

> **Tip**: Users can also paste their own API key directly in the sidebar, so you don't need to expose yours.

---

## 🔧 Prompt Engineering Techniques Used

### 1. System Persona
Sets Gemini's role, tone, and hard constraints before any user content.

### 2. Few-Shot Examples
Three worked examples (relevant answer, multi-source answer, "not found" case)
teach the model the exact output format expected.

### 3. Chain-of-Thought (THOUGHT → ANSWER)
Forces the model to reason about retrieved chunks before writing the answer,
reducing hallucinations on ambiguous queries.

### 4. Context Injection
Retrieved chunks are formatted with numbered labels and source metadata,
giving the model a structured, scannable knowledge base per query.

### 5. Citation Forcing
The prompt explicitly instructs the model to end every factual claim with
`[Source: <name>]`, making answers auditable.

### 6. Hallucination Guard
A hard rule ("NEVER hallucinate facts not present in the context") is
repeated both in the system prompt and the per-query instruction block.

---

## 🔄 Scaling Up

| Need | Swap to |
|---|---|
| Better embeddings | `sentence-transformers` + FAISS |
| Persistent index | ChromaDB or Pinecone |
| Larger context | `gemini-1.5-pro` (1M token window) |
| Async scraping | `httpx` + `asyncio` |
| Auth | Streamlit Authenticator |

---

## 📄 License

MIT — free to use and modify.
