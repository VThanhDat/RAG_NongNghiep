"""
BM25-style sparse embeddings for hybrid retrieval.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25Embedder:
    """
    BM25 sparse embedding using rank-bm25 when available.

    Falls back to a local TF-IDF style sparse vector if rank-bm25 is not
    installed, so indexing does not fail in minimal local environments.
    """

    def __init__(self):
        self._bm25 = None
        self._idf: dict[str, float] | None = None

    def fit(self, corpus: list[str]) -> "BM25Embedder":
        """Build IDF statistics from the corpus. Must be called before embedding."""
        tokenized = [self._tokenize(t) for t in corpus]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            doc_count = max(1, len(tokenized))
            df: Counter[str] = Counter()
            for tokens in tokenized:
                df.update(set(tokens))
            self._idf = {
                token: math.log((doc_count + 1) / (freq + 0.5)) + 1.0
                for token, freq in df.items()
            }
        else:
            self._bm25 = BM25Okapi(tokenized)
        return self

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        """Return a sparse vector per document."""
        return [self._vector(t) for t in texts]

    def embed_query(self, query: str) -> dict[str, float]:
        """Return a sparse vector for one query string."""
        return self._vector(query)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _vector(self, text: str) -> dict[str, float]:
        if self._bm25 is None and self._idf is None:
            raise RuntimeError("Call .fit(corpus_texts) before embedding.")
        tokens = self._tokenize(text)
        scores: dict[str, float] = {}
        if self._bm25 is not None:
            for token in set(tokens):
                score = float(self._bm25.get_scores([token]).max())
                if score > 0:
                    scores[token] = score
            return scores

        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        for token, count in counts.items():
            score = (count / total) * float((self._idf or {}).get(token, 0.0))
            if score > 0:
                scores[token] = score
        return scores
