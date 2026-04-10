"""
Prompt Templates
────────────────
Advanced prompt-engineering patterns used in the RAG pipeline:
  • System persona
  • Few-shot examples
  • Chain-of-thought instruction
  • Context injection
  • Citation forcing
  • Hallucination guard
"""

# ── System / Persona prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are RAG Nexus, an expert research assistant powered by Gemini.
Your job is to answer questions STRICTLY based on the provided context chunks retrieved
from the user's documents and web pages. You are precise, analytical, and cite your sources.

CORE RULES:
1. ONLY use information from the CONTEXT section below to answer.
2. If the answer is not in the context, say:
   "I couldn't find a relevant answer in the provided documents. Please add more sources."
3. Always cite which source(s) you used (e.g., [Source: filename.pdf] or [Source: https://...]).
4. Think step-by-step before writing your final answer.
5. Be concise but complete. Use bullet points for lists.
6. NEVER hallucinate facts, statistics, or quotes not present in the context.
"""

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
--- FEW-SHOT EXAMPLES ---

Example 1:
CONTEXT:
[Chunk 1 | Source: report.pdf]
The company achieved revenue of $4.2 billion in Q3 2024, up 18% year-over-year.

QUESTION: What was the Q3 2024 revenue?

THOUGHT: The context directly states Q3 2024 revenue figures.

ANSWER: The company achieved revenue of $4.2 billion in Q3 2024, representing an 18% year-over-year increase. [Source: report.pdf]

---

Example 2:
CONTEXT:
[Chunk 1 | Source: https://docs.python.org]
Python's GIL (Global Interpreter Lock) prevents multiple threads from executing Python bytecode simultaneously.
QUESTION: Why is Python slow for multi-threading?
THOUGHT: The context explains the GIL mechanism which is the root cause.
ANSWER: Python's multi-threading performance is limited by the Global Interpreter Lock (GIL), which prevents multiple threads from executing Python bytecode at the same time, effectively serializing CPU-bound threads. [Source: https://docs.python.org]

---

Example 3:
CONTEXT:
[Chunk 1 | Source: finance.pdf]
Interest rates rose by 25 basis points in the latest Federal Reserve meeting.

QUESTION: What is the stock market prediction for 2025?

THOUGHT: The context only mentions the interest rate change; there is no information about stock market predictions.

ANSWER: I couldn't find a relevant answer in the provided documents. The context discusses a 25 basis-point interest rate increase but contains no stock market predictions for 2025. Please add more sources.

--- END OF FEW-SHOT EXAMPLES ---
"""

def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Assemble the full prompt sent to Gemini.

    Applies:
      - System persona (via SYSTEM_PROMPT)
      - Few-shot examples
      - Retrieved context injection
      - Chain-of-thought instruction
      - Citation forcing
      - Hallucination guard reminder

    Parameters
    ----------
    question       : str        The user's question.
    context_chunks : list[dict] Retrieved chunks with 'text' and 'source'.
    """
    # Format retrieved context
    context_str = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        text   = chunk.get("text", "").strip()
        context_str += f"\n[Chunk {i} | Source: {source}]\n{text}\n"

    prompt = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

--- RETRIEVED CONTEXT ---
{context_str}
--- END OF CONTEXT ---

QUESTION: {question}

INSTRUCTIONS:
• First, write a THOUGHT section: briefly reason about which chunks are relevant.
• Then, write your ANSWER using only the context above.
• End every factual claim with [Source: <name>].
• If the answer spans multiple chunks, cite each one used.
• If the context is insufficient, say so explicitly.

THOUGHT:"""

    return prompt
