"""
Web Scraper
───────────
Fetches and cleans web pages, then chunks the content for RAG.
"""
from __future__ import annotations
import re
from typing import List
import requests
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

_UNWANTED_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "noscript", "svg", "iframe",
]


def _clean_html(html: str) -> str:
    """Parse HTML and return clean readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in _UNWANTED_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> List[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def scrape_urls(
    urls: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    timeout: int = 15,
) -> List[dict]:
    
    all_chunks: List[dict] = []
    for url in urls:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            text = _clean_html(resp.text)
            if len(text) < 100:
                continue
            chunks = _chunk_text(text, source=url, chunk_size=chunk_size, overlap=chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WebScraper] Failed to fetch {url}: {e}")
    return all_chunks
