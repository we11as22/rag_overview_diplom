"""Hybrid retrievers: linear weighted + RRF fusion.

Key optimisation: for a given (query, bm25, vector) triple the raw ranked
lists from BM25 and vector are identical regardless of alpha / RRF constant.
Both HybridRetriever and HybridRRFRetriever therefore share a single async
helper _fetch_candidates() that returns the two raw lists and caches nothing
itself — caching is done one level up in run_eval.py where we pre-compute
(bm25_raw, vec_raw) once per query and pass them directly via
*_from_cached_lists class methods.
"""
from __future__ import annotations

import asyncio
from typing import List, Tuple

from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever

RawList = List[Tuple[str, float]]


# ---------------------------------------------------------------------------
# Shared fetch helper
# ---------------------------------------------------------------------------

async def _fetch_candidates(
    bm25: BM25Retriever,
    vector: VectorRetriever,
    query: str,
    candidate_k: int,
) -> tuple[RawList, RawList]:
    """Fetch BM25 and vector candidate lists concurrently."""
    bm25_task = asyncio.to_thread(bm25.retrieve, query, candidate_k)
    vec_task = vector.aretrieve(query, top_k=candidate_k)
    return await asyncio.gather(bm25_task, vec_task)


# ---------------------------------------------------------------------------
# Score normalisation helpers
# ---------------------------------------------------------------------------

def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        # All equal → map to 0.0 (no signal) rather than 1.0
        return {k: 0.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


def _rrf_scores(ranked: RawList, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion score: 1 / (k + rank), 1-based."""
    return {doc_id: 1.0 / (k + rank) for rank, (doc_id, _) in enumerate(ranked, start=1)}


# ---------------------------------------------------------------------------
# Linear weighted hybrid
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Linear combination: alpha * vector_score + (1-alpha) * bm25_score.

    alpha=0.0 → pure BM25, alpha=1.0 → pure vector.
    """

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.bm25 = bm25
        self.vector = vector
        self.alpha = alpha

    # -- called from run_eval with pre-fetched lists (avoids re-embedding) --
    @staticmethod
    def fuse(
        bm25_raw: RawList,
        vec_raw: RawList,
        alpha: float,
        top_k: int,
    ) -> RawList:
        """Pure fusion from already-fetched lists. No IO."""
        bm25_norm = _min_max_normalize({d: s for d, s in bm25_raw})
        vec_norm  = _min_max_normalize({d: s for d, s in vec_raw})
        all_ids = set(bm25_norm) | set(vec_norm)
        combined = {
            doc_id: alpha * vec_norm.get(doc_id, 0.0) + (1.0 - alpha) * bm25_norm.get(doc_id, 0.0)
            for doc_id in all_ids
        }
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # -- standalone retrieve (used when called outside run_eval) --
    async def aretrieve(self, query: str, top_k: int = 10) -> RawList:
        candidate_k = min(top_k * 5, 200)
        bm25_raw, vec_raw = await _fetch_candidates(self.bm25, self.vector, query, candidate_k)
        return self.fuse(bm25_raw, vec_raw, self.alpha, top_k)

    def retrieve(self, query: str, top_k: int = 10) -> RawList:
        return asyncio.run(self.aretrieve(query, top_k))


# ---------------------------------------------------------------------------
# RRF hybrid
# ---------------------------------------------------------------------------

class HybridRRFRetriever:
    """Reciprocal Rank Fusion of BM25 and vector ranked lists.

    score = rrf_weight * rrf(vector_rank) + (1-rrf_weight) * rrf(bm25_rank)
    rrf_k controls rank discount: score(rank) = 1 / (rrf_k + rank).
    Standard default is rrf_k=60.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        rrf_k: int = 60,
        rrf_weight: float = 0.5,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight

    @staticmethod
    def fuse(
        bm25_raw: RawList,
        vec_raw: RawList,
        rrf_k: int,
        rrf_weight: float,
        top_k: int,
    ) -> RawList:
        """Pure RRF fusion from already-fetched lists. No IO."""
        bm25_rrf = _rrf_scores(bm25_raw, k=rrf_k)
        vec_rrf  = _rrf_scores(vec_raw,  k=rrf_k)
        all_ids = set(bm25_rrf) | set(vec_rrf)
        combined = {
            doc_id: rrf_weight * vec_rrf.get(doc_id, 0.0)
                    + (1.0 - rrf_weight) * bm25_rrf.get(doc_id, 0.0)
            for doc_id in all_ids
        }
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    async def aretrieve(self, query: str, top_k: int = 10) -> RawList:
        candidate_k = min(top_k * 5, 200)
        bm25_raw, vec_raw = await _fetch_candidates(self.bm25, self.vector, query, candidate_k)
        return self.fuse(bm25_raw, vec_raw, self.rrf_k, self.rrf_weight, top_k)

    def retrieve(self, query: str, top_k: int = 10) -> RawList:
        return asyncio.run(self.aretrieve(query, top_k))
