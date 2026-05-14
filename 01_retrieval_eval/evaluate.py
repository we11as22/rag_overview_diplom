"""Retrieval evaluation metrics — supports multiple k values."""
from __future__ import annotations

import asyncio
from typing import Any, List


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def mrr_at_k(retrieved_lists: List[List[str]], relevant_lists: List[List[str]], k: int) -> float:
    total = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant_set:
                total += 1.0 / rank
                break
    return total / len(retrieved_lists) if retrieved_lists else 0.0


def recall_at_k(retrieved_lists: List[List[str]], relevant_lists: List[List[str]], k: int) -> float:
    total = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        if not relevant:
            continue
        total += len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))
    return total / len(retrieved_lists) if retrieved_lists else 0.0


def precision_at_k(retrieved_lists: List[List[str]], relevant_lists: List[List[str]], k: int) -> float:
    total = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        top = retrieved[:k]
        if not top:
            continue
        total += len([d for d in top if d in set(relevant)]) / len(top)
    return total / len(retrieved_lists) if retrieved_lists else 0.0


def _metrics_for_k(retrieved_lists: List[List[str]], relevant_lists: List[List[str]], k: int) -> dict:
    return {
        f"mrr@{k}":       round(mrr_at_k(retrieved_lists, relevant_lists, k), 4),
        f"recall@{k}":    round(recall_at_k(retrieved_lists, relevant_lists, k), 4),
        f"precision@{k}": round(precision_at_k(retrieved_lists, relevant_lists, k), 4),
    }


def compute_all_metrics(
    retrieved_lists: List[List[str]],
    relevant_lists: List[List[str]],
    ks: List[int],
    n_queries: int,
) -> dict:
    """Compute metrics for all k values at once (single pass over the lists)."""
    result: dict = {"n_queries": n_queries}
    for k in ks:
        result.update(_metrics_for_k(retrieved_lists, relevant_lists, k))
    return result


# ---------------------------------------------------------------------------
# Async evaluation — generic retriever path
# ---------------------------------------------------------------------------

async def aevaluate_retriever(
    retriever: Any,
    qa_data: List[dict],
    ks: List[int],
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    """Evaluate a retriever over all QA pairs for all k values.

    Works with any retriever that exposes:
      aretrieve(query, top_k) or retrieve(query, top_k)

    top_k used for retrieval = max(ks) so we get enough candidates for all k.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(16)

    max_k = max(ks)

    async def _one_query(item: dict) -> tuple[List[str], List[str]]:
        query = item["question"]
        relevant = item.get("article_ids") or []
        async with semaphore:
            if hasattr(retriever, "aretrieve"):
                hits = await retriever.aretrieve(query, top_k=max_k)
            else:
                hits = await asyncio.to_thread(retriever.retrieve, query, max_k)
        return [doc_id for doc_id, _ in hits], relevant

    pairs = await asyncio.gather(*[_one_query(item) for item in qa_data])
    retrieved_lists = [p[0] for p in pairs]
    relevant_lists  = [p[1] for p in pairs]

    return compute_all_metrics(retrieved_lists, relevant_lists, ks, len(qa_data))


# ---------------------------------------------------------------------------
# Optimised hybrid evaluation — reuses cached (bm25_raw, vec_raw) per query
# ---------------------------------------------------------------------------

async def aevaluate_hybrid_batch(
    bm25_retriever,
    vector_retriever,
    qa_data: List[dict],
    ks: List[int],
    alphas: List[float],
    rrf_weights: List[float],
    rrf_k: int,
    semaphore: asyncio.Semaphore | None = None,
) -> List[dict]:
    """Evaluate all hybrid variants (alpha sweep + RRF sweep) in one pass.

    For each query we fetch (bm25_raw, vec_raw) ONCE and then run all fusion
    variants on the cached lists — zero extra Ollama calls.

    Returns a list of result dicts, one per (method, alpha/rrf_weight).
    """
    from retrieval.hybrid_retriever import (
        _fetch_candidates, HybridRetriever, HybridRRFRetriever
    )

    if semaphore is None:
        semaphore = asyncio.Semaphore(16)

    max_k = max(ks)
    candidate_k = min(max_k * 5, 200)

    # Step 1: fetch raw lists for all queries concurrently
    async def _fetch_one(item: dict):
        query = item["question"]
        relevant = item.get("article_ids") or []
        async with semaphore:
            bm25_raw, vec_raw = await _fetch_candidates(
                bm25_retriever, vector_retriever, query, candidate_k
            )
        return bm25_raw, vec_raw, relevant

    fetched = await asyncio.gather(*[_fetch_one(item) for item in qa_data])

    # Step 2: for each variant, fuse all queries (pure CPU — no IO)
    results = []

    for alpha in alphas:
        retrieved_lists, relevant_lists = [], []
        for bm25_raw, vec_raw, relevant in fetched:
            fused = HybridRetriever.fuse(bm25_raw, vec_raw, alpha, max_k)
            retrieved_lists.append([d for d, _ in fused])
            relevant_lists.append(relevant)
        metrics = compute_all_metrics(retrieved_lists, relevant_lists, ks, len(qa_data))
        results.append({"fusion": "linear", "alpha": alpha, **metrics})

    for rrf_w in rrf_weights:
        retrieved_lists, relevant_lists = [], []
        for bm25_raw, vec_raw, relevant in fetched:
            fused = HybridRRFRetriever.fuse(bm25_raw, vec_raw, rrf_k, rrf_w, max_k)
            retrieved_lists.append([d for d, _ in fused])
            relevant_lists.append(relevant)
        metrics = compute_all_metrics(retrieved_lists, relevant_lists, ks, len(qa_data))
        results.append({"fusion": "rrf", "rrf_weight": rrf_w, "rrf_k": rrf_k, **metrics})

    return results
