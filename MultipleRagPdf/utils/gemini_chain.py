from __future__ import annotations
import os
import re
import time
from typing import List, Dict, Any

import google.generativeai as genai

from utils.vector_store import VectorStore
from utils.prompt_templates import build_rag_prompt, SYSTEM_PROMPT


class GeminiRAGChain:
    MODEL_NAME = "gemini-2.5-flash"   # free tier: 5 req/min, 500 req/day

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 4,
        temperature: float = 0.3,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self._vs = vector_store
        self._top_k = top_k
        self._temperature = temperature
        self._model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1500,
            ),
            system_instruction=system_prompt,
        )

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self, question: str) -> Dict[str, Any]:
        # 1. Retrieve
        chunks = self._vs.retrieve(question, top_k=self._top_k)
        if not chunks:
            return {
                "answer": "I couldn't find relevant context. Please add more documents or URLs.",
                "sources": [],
                "thought": "",
                "chunks": [],
            }
        # 2. Build prompt
        prompt = build_rag_prompt(question, chunks)
        # 3. Call Gemini — auto-retry up to 3 times on 429 rate-limit
        raw = ""
        last_error = ""
        for attempt in range(4):
            try:
                response = self._model.generate_content(prompt)
                raw = response.text or ""
                break                        # success — exit retry loop
            except Exception as e:
                last_error = str(e)
                if "429" in last_error and attempt < 3:
                    wait_seconds = 35 * (attempt + 1)   # 35s -> 70s -> 105s
                    try:
                        import streamlit as st
                        with st.spinner(
                            f"Rate limit reached - waiting {wait_seconds}s "
                            f"then retrying ({attempt + 1}/3)..."
                        ):
                            time.sleep(wait_seconds)
                    except Exception:
                        time.sleep(wait_seconds)
                else:
                    return {
                        "answer": f"Gemini API error: {last_error}",
                        "sources": [],
                        "thought": "",
                        "chunks": chunks,
                    }

        if not raw:
            return {
                "answer": f"Gemini API error after retries: {last_error}",
                "sources": [],
                "thought": "",
                "chunks": chunks,
            }

        # 4. Parse THOUGHT / ANSWER sections
        thought, answer = self._parse_response(raw)

        # 5. Extract cited sources
        sources = self._extract_sources(answer, chunks)

        return {
            "answer": answer.strip(),
            "sources": sources,
            "thought": thought.strip(),
            "chunks": chunks,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str]:
        """Split raw Gemini output into (thought, answer)."""
        answer_match = re.search(r"ANSWER\s*[:]+\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
        thought_match = re.search(r"THOUGHT\s*[:]+\s*(.*?)(?=ANSWER|$)", raw, re.DOTALL | re.IGNORECASE)

        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            lines = raw.strip().split("\n")
            answer = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw.strip()

        thought = thought_match.group(1).strip() if thought_match else ""
        return thought, answer

    @staticmethod
    def _extract_sources(answer: str, chunks: List[dict]) -> List[str]:
        """Return unique source labels cited in the answer, or fallback to retrieved chunks."""
        cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer, re.IGNORECASE)
        if cited:
            return list(dict.fromkeys(s.strip() for s in cited))
        seen = set()
        sources = []
        for c in chunks:
            src = c.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources