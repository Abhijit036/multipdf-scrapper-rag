from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """In-memory TF-IDF retriever over text chunks."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None          # sparse TF-IDF matrix
        self._docs: List[dict] = []  # {"text": ..., "source": ...}

    def build(self, docs: List[dict]) -> None:
        """
        Index a list of chunk dicts.

        Parameters
        ----------
        docs : List[dict]
            Each dict must have 'text' and 'source' keys.
        """
        if not docs:
            raise ValueError("No documents provided to index.")
        self._docs = docs
        texts = [d["text"] for d in docs]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30_000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 4) -> List[dict]:
        """
        Return top-k most relevant chunks for *query*.

        Returns
        -------
        List[dict]   Sorted by relevance (best first).
                     Each dict: {'text': str, 'source': str, 'score': float}
        """
        if self._vectorizer is None or self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = dict(self._docs[idx])
                doc["score"] = float(scores[idx])
                results.append(doc)
        return results
