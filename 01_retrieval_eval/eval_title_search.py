"""Оценка поиска по заголовкам статей (title search evaluation).

Задача: по вопросу из QA-датасета найти заголовок правильной статьи.
Это отдельная от chunk-поиска задача — агент может сначала искать
по заголовкам чтобы понять что вообще есть по теме.

Метрика: Hit@K — правильная статья попала в топ-K результатов поиска.
Дополнительно: MRR@K, Recall@K.

Методы:
  - BM25 по заголовкам
  - Vector (Ollama) по заголовкам
  - Hybrid (BM25 + vector) по заголовкам с разными alpha

Только single-article QA пары (148 из 200).

Usage:
    python eval_title_search.py
    python eval_title_search.py --dry-run
    python eval_title_search.py --models embeddinggemma,bge-m3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config
from embeddings import OllamaEmbedder
from evaluate import filter_single_article, compute_all_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> Tuple[List[dict], List[dict]]:
    corpus_path = config.DATA_DIR / "corpus.jsonl"
    qa_path     = config.DATA_DIR / "qa_expertwritten.jsonl"

    for p in (corpus_path, qa_path):
        if not p.exists():
            sys.exit(f"[ERROR] Not found: {p}. Run `python download_data.py` first.")

    with corpus_path.open() as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    with qa_path.open() as f:
        qa = [json.loads(l) for l in f if l.strip()]

    qa_single = filter_single_article(qa)
    print(f"Corpus: {len(corpus)} docs  |  QA (single-article): {len(qa_single)}/200")
    return corpus, qa_single


# ---------------------------------------------------------------------------
# BM25 title retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class TitleBM25:
    def __init__(self, corpus: List[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._ids = [d["id"] for d in corpus]
        self._bm25 = BM25Okapi([_tokenize(d.get("title", "")) for d in corpus])

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# Vector title retriever (Ollama)
# ---------------------------------------------------------------------------

class TitleVectorRetriever:
    """Хранит эмбеддинги заголовков в памяти (6221 * 768 float32 ≈ 18 MB)."""

    def __init__(self, corpus: List[dict], embedder: OllamaEmbedder) -> None:
        self._ids = [d["id"] for d in corpus]
        self._embedder = embedder
        self._vectors = None

    async def build_from_corpus(self, corpus: List[dict]) -> None:
        import numpy as np
        print(f"  Embedding {len(corpus)} titles...", end="", flush=True)
        titles = [d.get("title", "") for d in corpus]
        vecs = await self._embedder.aembed(titles, is_query=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Нулевые векторы (пустые заголовки) заменяем единицей чтобы не получить nan
        norms = np.where(norms == 0, 1.0, norms)
        self._vectors = (vecs / norms).astype(np.float64)
        # Проверяем что нет nan/inf после нормализации
        bad = np.isnan(self._vectors).any(axis=1) | np.isinf(self._vectors).any(axis=1)
        if bad.any():
            self._vectors[bad] = 0.0  # нулевой вектор → никогда не попадёт в топ
        print(f" done. Shape: {self._vectors.shape}, bad_rows={bad.sum()}")

    def retrieve(self, query_vec, top_k: int) -> List[Tuple[str, float]]:
        import numpy as np
        q = query_vec.astype(np.float64)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = np.dot(self._vectors, q)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top]

    async def aretrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q_vec = await self._embedder.aembed_query(query)
        return self.retrieve(q_vec, top_k)


# ---------------------------------------------------------------------------
# Hybrid title retriever
# ---------------------------------------------------------------------------

def _min_max(scores: dict) -> dict:
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class TitleHybrid:
    def __init__(self, bm25: TitleBM25, vector: TitleVectorRetriever, alpha: float):
        self.bm25 = bm25
        self.vector = vector
        self.alpha = alpha

    async def aretrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        candidate_k = min(top_k * 5, 500)
        bm25_raw  = self.bm25.retrieve(query, candidate_k)
        vec_raw   = await self.vector.aretrieve(query, candidate_k)

        bm25_norm = _min_max({d: s for d, s in bm25_raw})
        vec_norm  = _min_max({d: s for d, s in vec_raw})

        all_ids = set(bm25_norm) | set(vec_norm)
        combined = {
            did: self.alpha * vec_norm.get(did, 0.0) + (1 - self.alpha) * bm25_norm.get(did, 0.0)
            for did in all_ids
        }
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

async def evaluate_title_retriever(retriever, qa_data: List[dict], ks: List[int]) -> dict:
    """Оценить title-retriever: для каждого вопроса находим правильную статью по заголовку."""
    max_k = max(ks)
    sem = asyncio.Semaphore(16)

    async def _one(item: dict) -> Tuple[List[str], List[str]]:
        query    = item["question"]
        relevant = item["article_ids"]   # уже single-article
        async with sem:
            if hasattr(retriever, "aretrieve"):
                hits = await retriever.aretrieve(query, top_k=max_k)
            else:
                hits = await asyncio.to_thread(retriever.retrieve, query, max_k)
        return [d for d, _ in hits], relevant

    pairs = await asyncio.gather(*[_one(item) for item in qa_data])
    retrieved_lists = [p[0] for p in pairs]
    relevant_lists  = [p[1] for p in pairs]

    # Hit@K — хотя бы одна релевантная статья попала в топ-K
    def hit_at_k(k: int) -> float:
        hits = sum(
            1 for ret, rel in zip(retrieved_lists, relevant_lists)
            if set(ret[:k]) & set(rel)
        )
        return round(hits / len(qa_data), 4)

    metrics = compute_all_metrics(retrieved_lists, relevant_lists, ks, len(qa_data))
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(k)
    return metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _make_report(results: List[dict], ks: List[int]) -> str:
    primary = f"mrr@{max(ks)}"
    results_sorted = sorted(results, key=lambda r: r.get(primary, 0), reverse=True)

    k_headers = " | ".join(f"Hit@{k} MRR@{k}" for k in ks)
    lines = [
        "# Title Search Evaluation Report",
        "",
        f"Single-article QA pairs only (148/200)  |  k values: {ks}",
        "",
        "| Rank | Method | Model | Alpha |" + "".join(f" Hit@{k} | MRR@{k} |" for k in ks),
        "|------|--------|-------|-------|" + "".join("-------|-------|" for _ in ks),
    ]

    for rank, r in enumerate(results_sorted, 1):
        method = r["method"]
        model  = r.get("model") or "—"
        alpha  = f"{r['alpha']:.1f}" if r.get("alpha") is not None else "—"
        cells  = "".join(
            f" {r.get(f'hit@{k}', 0):.4f} | {r.get(f'mrr@{k}', 0):.4f} |"
            for k in ks
        )
        lines.append(f"| {rank} | {method} | {model} | {alpha} |{cells}")

    lines += ["", "## Best configurations per method", ""]
    for method in ("bm25_title", "vector_title", "hybrid_title"):
        bucket = [r for r in results_sorted if r["method"] == method]
        if not bucket:
            continue
        best = bucket[0]
        mrr  = best.get(primary, 0)
        hit  = best.get(f"hit@{max(ks)}", 0)
        model = best.get("model") or "—"
        alpha = f", α={best['alpha']:.1f}" if best.get("alpha") is not None else ""
        lines.append(f"- **{method}** ({model}{alpha}): Hit@{max(ks)}={hit:.4f}, MRR@{max(ks)}={mrr:.4f}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(dry_run: bool, models_override: List[str] | None) -> None:
    corpus, qa_data = load_data()
    ks = config.EVAL_KS

    if dry_run:
        qa_data = qa_data[:5]
        print(f"[dry-run] 5 queries")

    models = models_override or config.OLLAMA_EMBED_MODELS
    alphas = config.HYBRID_ALPHAS
    all_results: List[dict] = []

    # --- BM25 title (один раз) ---
    print("\n[BM25 title] Building index...")
    t0 = time.time()
    bm25 = TitleBM25(corpus)
    print(f"[BM25 title] Done in {time.time()-t0:.1f}s")

    print("[BM25 title] Evaluating...")
    t0 = time.time()

    async def _bm25_aretrieve(query, top_k):
        return await asyncio.to_thread(bm25.retrieve, query, top_k)

    class _BM25Wrapper:
        async def aretrieve(self, q, top_k):
            return await asyncio.to_thread(bm25.retrieve, q, top_k)

    bm25_metrics = await evaluate_title_retriever(_BM25Wrapper(), qa_data, ks)
    all_results.append({"method": "bm25_title", "model": None, "alpha": None, **bm25_metrics})
    print(f"[BM25 title] MRR@{max(ks)}={bm25_metrics.get(f'mrr@{max(ks)}'):.4f}  Hit@{max(ks)}={bm25_metrics.get(f'hit@{max(ks)}'):.4f}  ({time.time()-t0:.1f}s)")

    # --- Per embedding model ---
    for model_name in models:
        print(f"\n{'─'*50}")
        print(f"Model: {model_name}")
        print(f"{'─'*50}")

        embedder = OllamaEmbedder(model=model_name, base_url=config.OLLAMA_BASE_URL, concurrency=8)
        vector = TitleVectorRetriever(corpus, embedder)
        await vector.build_from_corpus(corpus)

        # Vector title
        t0 = time.time()
        vec_metrics = await evaluate_title_retriever(vector, qa_data, ks)
        all_results.append({"method": "vector_title", "model": model_name, "alpha": None, **vec_metrics})
        print(f"[vector] MRR@{max(ks)}={vec_metrics.get(f'mrr@{max(ks)}'):.4f}  Hit@{max(ks)}={vec_metrics.get(f'hit@{max(ks)}'):.4f}  ({time.time()-t0:.1f}s)")

        # Hybrid title — все alpha сразу (результаты BM25 и vector уже есть в памяти)
        print(f"[hybrid] Sweeping {len(alphas)} alpha values...")
        t0 = time.time()

        sem = asyncio.Semaphore(16)

        async def _fetch_one(item):
            query = item["question"]
            relevant = item["article_ids"]
            candidate_k = min(max(ks) * 5, 500)
            async with sem:
                bm25_raw = await asyncio.to_thread(bm25.retrieve, query, candidate_k)
                vec_raw  = await vector.aretrieve(query, candidate_k)
            return bm25_raw, vec_raw, relevant

        fetched = await asyncio.gather(*[_fetch_one(item) for item in qa_data])

        for alpha in alphas:
            retrieved_lists, relevant_lists = [], []
            for bm25_raw, vec_raw, relevant in fetched:
                bm25_norm = _min_max({d: s for d, s in bm25_raw})
                vec_norm  = _min_max({d: s for d, s in vec_raw})
                all_ids   = set(bm25_norm) | set(vec_norm)
                combined  = sorted(
                    {did: alpha * vec_norm.get(did, 0.0) + (1-alpha) * bm25_norm.get(did, 0.0)
                     for did in all_ids}.items(),
                    key=lambda x: x[1], reverse=True
                )[:max(ks)]
                retrieved_lists.append([d for d, _ in combined])
                relevant_lists.append(relevant)

            metrics = compute_all_metrics(retrieved_lists, relevant_lists, ks, len(qa_data))
            for k in ks:
                hits = sum(1 for ret, rel in zip(retrieved_lists, relevant_lists) if set(ret[:k]) & set(rel))
                metrics[f"hit@{k}"] = round(hits / len(qa_data), 4)

            all_results.append({"method": "hybrid_title", "model": model_name, "alpha": alpha, **metrics})
            print(f"  α={alpha:.1f}  MRR@{max(ks)}={metrics.get(f'mrr@{max(ks)}'):.4f}  Hit@{max(ks)}={metrics.get(f'hit@{max(ks)}'):.4f}")

        print(f"[hybrid] done in {time.time()-t0:.1f}s")

    # Save
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.RESULTS_DIR / "title_search_results.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {json_path}")

    report = _make_report(all_results, ks)
    md_path = config.RESULTS_DIR / "title_search_report.md"
    with md_path.open("w") as f:
        f.write(report)
    print(f"Saved -> {md_path}")
    print()
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Title search evaluation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", help="Comma-separated model names (override .env)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run(dry_run=args.dry_run, models_override=models))


if __name__ == "__main__":
    main()
