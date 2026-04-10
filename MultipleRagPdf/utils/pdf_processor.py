"""
PDF Processor
─────────────
Extracts text from uploaded PDFs and splits into overlapping chunks.
"""
from __future__ import annotations
import io
from typing import List
import pypdf


def _chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> List[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def process_pdfs(
    uploaded_files,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Extract text from list of Streamlit UploadedFile objects (PDFs).

    Returns
    -------
    List[dict]  Each dict has keys: 'text', 'source'
    """
    all_chunks: List[dict] = []

    for uf in uploaded_files:
        raw = uf.read()
        reader = pypdf.PdfReader(io.BytesIO(raw))
        full_text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text_parts.append(f"[Page {page_num}]\n{page_text}")
        full_text = "\n\n".join(full_text_parts)
        chunks = _chunk_text(full_text, source=uf.name, chunk_size=chunk_size, overlap=chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
