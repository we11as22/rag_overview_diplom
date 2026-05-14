"""BM25 full-text retriever using rank-bm25."""
from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """BM25 retriever over a corpus of documents.

    Each document is expected to have fields: id, title, contents.
    The index is built over the concatenation of title + contents.
    """

    def __init__(self, corpus: List[dict]) -> None:
        self._doc_ids: List[str] = [doc["id"] for doc in corpus]
        tokenized = [_tokenize(f"{doc.get('title', '')} {doc.get('contents', '')}") for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, score) pairs for the given query."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Partial sort: get indices of top-k scores
        if top_k >= len(scores):
            top_indices = list(range(len(scores)))
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [(self._doc_ids[i], float(scores[i])) for i in top_indices]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
