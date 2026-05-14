"""LLM Chain-of-RAG retriever.

Pipeline per query
──────────────────
1. Hybrid search over TITLE index (top-N docs by title only)        → title_docs
2. LLM generates K follow-up sub-questions from original query
3. For each sub-question: hybrid search over CHUNK index (contents) → chunk_hits
4. Deduplicate all found doc IDs, collect full texts
5. LLM summarises all retrieved snippets given the original question
6. Return the union of all retrieved doc IDs (for MRR evaluation)
   plus the summary text (for downstream answer quality, not evaluated here)

For MRR/Recall the relevant set is article_ids from the QA pair, and we
score against the *union* of all doc IDs surfaced in steps 1 and 3.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import List, Tuple

import httpx

from embeddings import OllamaEmbedder

RawList = List[Tuple[str, float]]

# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 512    # chars
_CHUNK_OVERLAP = 64  # chars


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class _LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self._base = api_base.rstrip("/")
        self._key = api_key
        self._model = model

    async def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self._base}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Chunk index (separate ChromaDB collection per model)
# ---------------------------------------------------------------------------

class ChunkIndex:
    """Builds and queries a ChromaDB collection of document chunks."""

    def __init__(
        self,
        corpus: List[dict],
        embedder: OllamaEmbedder,
        persist_dir,
        force_rebuild: bool = False,
    ) -> None:
        import chromadb, re as _re
        from pathlib import Path

        self._embedder = embedder
        self._corpus_map = {doc["id"]: doc for doc in corpus}

        persist_dir = Path(persist_dir)
        coll_name_raw = f"chunks-{embedder.model}"
        coll_name = _re.sub(r"[^a-zA-Z0-9-]", "-", coll_name_raw)
        coll_name = _re.sub(r"-+", "-", coll_name).strip("-")[:63]

        client = chromadb.PersistentClient(path=str(persist_dir))
        collections = client.list_collections()
        existing = {c.name if hasattr(c, "name") else c for c in collections}

        if force_rebuild and coll_name in existing:
            client.delete_collection(coll_name)
            existing.discard(coll_name)

        self._coll = client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Deferred async build — don't call asyncio.run here (caller is async).
        # Also rebuild if collection exists but empty (interrupted previous build).
        populated = coll_name in existing and self._coll.count() > 0
        self._corpus_to_build = None if populated else corpus

    async def ensure_built(self) -> None:
        """Async build if needed. Must be awaited before first use."""
        if self._corpus_to_build is not None:
            await self._build(self._corpus_to_build)
            self._corpus_to_build = None

    async def _build(self, corpus: List[dict]) -> None:
        print(f"  [ChunkIndex] Building chunk index for '{self._embedder.model}'...")
        batch_ids, batch_texts, batch_meta = [], [], []
        chunk_count = 0

        for doc in corpus:
            doc_id = doc["id"]
            chunks = _chunk_text(doc.get("contents", ""))
            for ci, chunk in enumerate(chunks):
                cid = f"{doc_id}__chunk{ci}"
                batch_ids.append(cid)
                batch_texts.append(chunk)
                batch_meta.append({"doc_id": doc_id})
                chunk_count += 1

                if len(batch_ids) >= 64:
                    vecs = await self._embedder.aembed(batch_texts, is_query=False)
                    self._coll.upsert(ids=batch_ids, embeddings=vecs.tolist(), metadatas=batch_meta)
                    batch_ids, batch_texts, batch_meta = [], [], []

        if batch_ids:
            vecs = await self._embedder.aembed(batch_texts, is_query=False)
            self._coll.upsert(ids=batch_ids, embeddings=vecs.tolist(), metadatas=batch_meta)

        print(f"  [ChunkIndex] Done: {chunk_count} chunks from {len(corpus)} docs.")

    async def query(self, query: str, top_k: int = 10) -> List[str]:
        """Return list of unique doc_ids from top-k chunk hits."""
        q_vec = await self._embedder.aembed_query(query)
        results = self._coll.query(
            query_embeddings=[q_vec.tolist()],
            n_results=min(top_k, self._coll.count()),
            include=["metadatas"],
        )
        seen, doc_ids = set(), []
        for meta in results["metadatas"][0]:
            did = meta["doc_id"]
            if did not in seen:
                seen.add(did)
                doc_ids.append(did)
        return doc_ids


# ---------------------------------------------------------------------------
# Title-only BM25 index
# ---------------------------------------------------------------------------

class TitleBM25Retriever:
    """BM25 over titles only."""

    def __init__(self, corpus: List[dict]) -> None:
        import re
        from rank_bm25 import BM25Okapi

        self._ids = [doc["id"] for doc in corpus]
        tokenized = [re.findall(r"\w+", doc.get("title", "").lower()) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> RawList:
        import re
        tokens = re.findall(r"\w+", query.lower())
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Main Chain-of-RAG retriever
# ---------------------------------------------------------------------------

class ChainRAGRetriever:
    """LLM-driven chain retriever.

    Parameters
    ----------
    corpus          : list of corpus dicts (id, title, contents)
    title_bm25      : TitleBM25Retriever (BM25 over titles)
    chunk_index     : ChunkIndex for content-level chunk retrieval
    llm             : _LLMClient (OpenAI-compatible)
    title_top_n     : docs to pull in step 1 (title BM25 search)
    n_subquestions  : follow-up questions the LLM generates
    chunk_top_k     : chunk hits per sub-question
    """

    def __init__(
        self,
        corpus: List[dict],
        title_bm25: TitleBM25Retriever,
        chunk_index: ChunkIndex,
        llm: "_LLMClient",
        title_top_n: int = 5,
        n_subquestions: int = 3,
        chunk_top_k: int = 5,
    ) -> None:
        self._corpus_map = {doc["id"]: doc for doc in corpus}
        self._title_bm25 = title_bm25
        self._chunk_index = chunk_index
        self._llm = llm
        self._title_top_n = title_top_n
        self._n_subquestions = n_subquestions
        self._chunk_top_k = chunk_top_k

    # -- LLM calls --

    async def _generate_subquestions(self, question: str) -> List[str]:
        prompt = (
            f"You are a research assistant. Given the user question below, generate "
            f"{self._n_subquestions} specific follow-up sub-questions that would help find "
            f"all relevant information needed to answer it. "
            f"Return ONLY a JSON array of strings, no explanation.\n\n"
            f"Question: {question}"
        )
        raw = await self._llm.chat([{"role": "user", "content": prompt}])
        try:
            # Extract JSON array even if wrapped in markdown
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        # Fallback: return lines that look like questions
        lines = [l.strip().lstrip("-•123456789. ") for l in raw.splitlines() if "?" in l]
        return lines[: self._n_subquestions] or [question]

    async def _summarise(self, question: str, snippets: List[str]) -> str:
        combined = "\n\n---\n\n".join(snippets[:10])  # cap to avoid huge context
        prompt = (
            f"Based on the following document excerpts, answer the question concisely.\n\n"
            f"Question: {question}\n\n"
            f"Documents:\n{combined}"
        )
        return await self._llm.chat([{"role": "user", "content": prompt}])

    # -- Retrieval pipeline --

    async def aretrieve(self, query: str, top_k: int = 10) -> RawList:
        """Returns list of (doc_id, score) for all docs surfaced during the chain.

        For MRR evaluation, 'top_k' controls how many IDs are returned.
        Score is a simple rank-based proxy (1 / rank).
        """
        # Step 1: title-level BM25 search
        title_hits = await asyncio.to_thread(
            self._title_bm25.retrieve, query, self._title_top_n
        )
        title_doc_ids = [did for did, _ in title_hits]

        # Step 2: generate sub-questions concurrently with nothing else yet
        subquestions = await self._generate_subquestions(query)

        # Step 3: chunk-level retrieval for each sub-question (concurrent)
        chunk_results = await asyncio.gather(*[
            self._chunk_index.query(sq, top_k=self._chunk_top_k)
            for sq in subquestions
        ])

        # Step 4: collect all unique doc IDs (title hits first, then chunk hits)
        seen: set[str] = set()
        ordered_ids: List[str] = []
        for did in title_doc_ids:
            if did not in seen:
                seen.add(did)
                ordered_ids.append(did)
        for chunk_docs in chunk_results:
            for did in chunk_docs:
                if did not in seen:
                    seen.add(did)
                    ordered_ids.append(did)

        # Step 5: summarise (fire-and-forget for eval purposes — we only need IDs for MRR)
        # We still call it so the pipeline is complete; result is stored on the retriever
        # but not used in the returned ranked list.
        snippets = [
            self._corpus_map[did].get("contents", "")[:800]
            for did in ordered_ids
            if did in self._corpus_map
        ]
        # Run summarisation in background — don't block MRR scoring
        asyncio.create_task(self._summarise(query, snippets))

        # Return as ranked list (rank = position in ordered_ids)
        ranked: RawList = [(did, 1.0 / (rank + 1)) for rank, did in enumerate(ordered_ids)]
        return ranked[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> RawList:
        return asyncio.run(self.aretrieve(query, top_k))
