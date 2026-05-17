"""Main evaluation orchestrator (async).

Usage:
    python run_eval.py                  # full run
    python run_eval.py --dry-run        # first 5 queries only
    python run_eval.py --force-rebuild  # rebuild all vector indices
    python run_eval.py --skip-chain     # skip LLM chain-of-RAG (no LLM API needed)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

import config
from embeddings import OllamaEmbedder
from evaluate import aevaluate_retriever, aevaluate_hybrid_batch, filter_single_article
from retrieval import BM25Retriever, VectorRetriever
from retrieval.chain_rag_retriever import (
    ChainRAGRetriever, ChunkIndex, TitleBM25Retriever, _LLMClient,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus() -> List[dict]:
    path = config.DATA_DIR / "corpus.jsonl"
    if not path.exists():
        sys.exit(f"[ERROR] Corpus not found at {path}. Run `python download_data.py` first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qa(dry_run: bool = False) -> List[dict]:
    path = config.DATA_DIR / "qa_expertwritten.jsonl"
    if not path.exists():
        sys.exit(f"[ERROR] QA data not found at {path}. Run `python download_data.py` first.")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Оцениваем только на single-article парах — однозначная метрика
    rows = filter_single_article(rows)
    print(f"[filter] Using {len(rows)} single-article QA pairs (out of 200 total)")

    if dry_run:
        print(f"[dry-run] Using first 5 queries")
        rows = rows[:5]
    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _metric_cols(ks: List[int]) -> List[str]:
    cols = []
    for k in ks:
        cols += [f"mrr@{k}", f"recall@{k}", f"precision@{k}"]
    return cols


def _make_report(results: List[dict], ks: List[int], primary_k: int) -> str:
    primary_mrr = f"mrr@{primary_k}"
    results_sorted = sorted(results, key=lambda r: r.get(primary_mrr, 0), reverse=True)

    metric_cols = _metric_cols(ks)

    # Header
    header_parts = ["| Rank | Method | Embed Model | Fusion | Param |"]
    sep_parts    = ["|------|--------|-------------|--------|-------|"]
    for k in ks:
        header_parts[0] += f" MRR@{k} | Recall@{k} | P@{k} |"
        sep_parts[0]    += "--------|-----------|------|"

    lines = [
        "# RAG Retrieval Evaluation Report",
        "",
        f"Primary sort metric: **MRR@{primary_k}**  |  Queries: **{results[0]['n_queries'] if results else 0}**  |  k values: {ks}",
        "",
        header_parts[0],
        sep_parts[0],
    ]

    for rank, r in enumerate(results_sorted, start=1):
        method = r.get("method", "—")
        model  = r.get("embed_model") or "—"
        fusion = r.get("fusion", "—")

        # Param column: alpha for linear, rrf_weight for rrf, — for others
        if r.get("fusion") == "linear" and r.get("alpha") is not None:
            param = f"α={r['alpha']:.1f}"
        elif r.get("fusion") == "rrf":
            param = f"w={r.get('rrf_weight', '?')}, k={r.get('rrf_k', '?')}"
        else:
            param = "—"

        row = f"| {rank} | {method} | {model} | {fusion} | {param} |"
        for k in ks:
            mrr  = r.get(f"mrr@{k}", 0)
            rec  = r.get(f"recall@{k}", 0)
            prec = r.get(f"precision@{k}", 0)
            row += f" {mrr:.4f} | {rec:.4f} | {prec:.4f} |"
        lines.append(row)

    # Best per method
    lines += ["", "## Best per method", ""]
    for method in ("bm25", "vector", "hybrid_linear", "hybrid_rrf", "chain_rag"):
        bucket = [r for r in results_sorted if r.get("method") == method]
        if not bucket:
            continue
        best = bucket[0]
        mrr = best.get(primary_mrr, 0)
        model = best.get("embed_model") or "—"
        if best.get("fusion") == "linear":
            detail = f"α={best.get('alpha', '?'):.1f}"
        elif best.get("fusion") == "rrf":
            detail = f"rrf_w={best.get('rrf_weight', '?')}"
        else:
            detail = ""
        lines.append(f"- **{method}** ({model} {detail}): MRR@{primary_k} = **{mrr:.4f}**")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async main loop
# ---------------------------------------------------------------------------

async def run_all_experiments(
    dry_run: bool = False,
    force_rebuild: bool = False,
    skip_chain: bool = False,
) -> List[dict]:
    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)

    corpus = load_corpus()
    qa_data = load_qa(dry_run=dry_run)
    ks = config.EVAL_KS

    print(f"Corpus: {len(corpus)} documents")
    print(f"QA pairs: {len(qa_data)}")
    print(f"Embedding models: {config.OLLAMA_EMBED_MODELS}")
    print(f"Eval k values: {ks}  |  Hybrid alphas: {config.HYBRID_ALPHAS}")
    print(f"RRF weights: {config.RRF_WEIGHTS}  |  RRF_K: {config.RRF_K}")
    print()

    # Semaphore for Ollama/LLM calls during evaluation queries
    eval_sem = asyncio.Semaphore(16)

    all_results: List[dict] = []

    # ── BM25 ──────────────────────────────────────────────────────────────
    print("[BM25] Building index...")
    t0 = time.time()
    bm25 = BM25Retriever(corpus)
    print(f"[BM25] Done in {time.time()-t0:.1f}s")

    print("[BM25] Evaluating...")
    t0 = time.time()
    bm25_metrics = await aevaluate_retriever(bm25, qa_data, ks=ks, semaphore=eval_sem)
    all_results.append({"method": "bm25", "embed_model": None, "fusion": "—", **bm25_metrics})
    print(f"[BM25] MRR@{config.TOP_K} = {bm25_metrics.get(f'mrr@{config.TOP_K}', '?'):.4f}  ({time.time()-t0:.1f}s)")

    # ── Per embedding model ───────────────────────────────────────────────
    for model_name in config.OLLAMA_EMBED_MODELS:
        print()
        print(f"{'─' * 55}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 55}")

        embedder = OllamaEmbedder(
            model=model_name,
            base_url=config.OLLAMA_BASE_URL,
            concurrency=8,
        )
        chroma_path = config.CHROMA_DIR / model_name.replace(":", "_").replace("/", "_")

        # ── Vector index ──
        print(f"  Building / loading vector index...")
        t0 = time.time()
        vector = VectorRetriever(
            corpus=corpus,
            embedder=embedder,
            persist_dir=chroma_path,
            force_rebuild=force_rebuild,
        )
        await vector.ensure_built()
        print(f"  Vector index ready in {time.time()-t0:.1f}s")

        # ── Vector eval ──
        print(f"  [vector] Evaluating...")
        t0 = time.time()
        vec_metrics = await aevaluate_retriever(vector, qa_data, ks=ks, semaphore=eval_sem)
        all_results.append({"method": "vector", "embed_model": model_name, "fusion": "dense", **vec_metrics})
        print(f"  [vector] MRR@{config.TOP_K} = {vec_metrics.get(f'mrr@{config.TOP_K}', '?'):.4f}  ({time.time()-t0:.1f}s)")

        # ── Hybrid batch eval (linear + RRF, single fetch per query) ──
        print(f"  [hybrid] Evaluating {len(config.HYBRID_ALPHAS)} linear + {len(config.RRF_WEIGHTS)} RRF variants...")
        t0 = time.time()
        hybrid_results = await aevaluate_hybrid_batch(
            bm25_retriever=bm25,
            vector_retriever=vector,
            qa_data=qa_data,
            ks=ks,
            alphas=config.HYBRID_ALPHAS,
            rrf_weights=config.RRF_WEIGHTS,
            rrf_k=config.RRF_K,
            semaphore=eval_sem,
        )
        for hr in hybrid_results:
            if hr.get("fusion") == "linear":
                method = "hybrid_linear"
            else:
                method = "hybrid_rrf"
            all_results.append({"method": method, "embed_model": model_name, **hr})
            tag = f"α={hr.get('alpha','?'):.1f}" if hr.get("fusion") == "linear" else f"rrf_w={hr.get('rrf_weight','?')}"
            print(f"    {tag}  MRR@{config.TOP_K}={hr.get(f'mrr@{config.TOP_K}', '?'):.4f}")

        print(f"  [hybrid] done in {time.time()-t0:.1f}s")

        # ── Chain-of-RAG (one per embedding model) ──
        if not skip_chain:
            print(f"  [chain-rag] Building chunk index...")
            t0 = time.time()
            chunk_chroma = config.CHROMA_DIR / ("chunks_" + model_name.replace(":", "_").replace("/", "_"))
            chunk_index = ChunkIndex(
                corpus=corpus,
                embedder=embedder,
                persist_dir=chunk_chroma,
                force_rebuild=force_rebuild,
            )
            await chunk_index.ensure_built()  # async build — no nested asyncio.run
            title_bm25 = TitleBM25Retriever(corpus)
            llm = _LLMClient(
                api_base=config.LLM_API_BASE,
                api_key=config.LLM_API_KEY,
                model=config.LLM_MODEL,
            )
            chain = ChainRAGRetriever(
                corpus=corpus,
                title_bm25=title_bm25,
                chunk_index=chunk_index,
                llm=llm,
                title_top_n=config.CHAIN_TITLE_TOP_N,
                n_subquestions=config.CHAIN_N_SUBQUESTIONS,
                chunk_top_k=config.CHAIN_CHUNK_TOP_K,
            )
            print(f"  [chain-rag] Chunk index ready in {time.time()-t0:.1f}s")

            print(f"  [chain-rag] Evaluating (LLM calls, may be slow)...")
            t0 = time.time()
            chain_metrics = await aevaluate_retriever(
                chain, qa_data, ks=ks, semaphore=asyncio.Semaphore(4)  # lower concurrency for LLM
            )
            all_results.append({"method": "chain_rag", "embed_model": model_name, "fusion": "chain", **chain_metrics})
            print(f"  [chain-rag] MRR@{config.TOP_K} = {chain_metrics.get(f'mrr@{config.TOP_K}', '?'):.4f}  ({time.time()-t0:.1f}s)")

    return all_results


def save_results(results: List[dict]) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = config.RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {json_path}")

    report = _make_report(results, ks=config.EVAL_KS, primary_k=config.TOP_K)
    md_path = config.RESULTS_DIR / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {md_path}")
    print()
    print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--dry-run",       action="store_true", help="First 5 queries only")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild all vector indices")
    parser.add_argument("--skip-chain",    action="store_true", help="Skip LLM chain-of-RAG")
    args = parser.parse_args()

    results = asyncio.run(run_all_experiments(
        dry_run=args.dry_run,
        force_rebuild=args.force_rebuild,
        skip_chain=args.skip_chain,
    ))
    save_results(results)


if __name__ == "__main__":
    main()
