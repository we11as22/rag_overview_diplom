"""Async Ollama-based text embeddings with per-model query formatting."""
from __future__ import annotations

import asyncio
from typing import List

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Query-time text formatters per model family
# ---------------------------------------------------------------------------
# Applied only when embedding queries (not corpus documents).
# Corpus documents are always embedded as-is.

def _format_query(text: str, model: str) -> str:
    """Apply model-specific query prefix for retrieval."""
    m = model.lower()
    if "embeddinggemma" in m or "gemma" in m:
        return f"task: search result | query: {text}"
    if "qwen3" in m or "qwen" in m:
        task_description = "Given a question, retrieve relevant documents that answer the question"
        return f"Instruct: {task_description}\nQuery:{text}"
    return text


def _format_doc(text: str, model: str) -> str:
    """Corpus documents — no prefix for any current model."""
    return text


# ---------------------------------------------------------------------------
# Async embedder
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Async Ollama embedder with a semaphore to cap concurrent requests.

    Parameters
    ----------
    model       : Ollama model name, e.g. "gemma3-embedding"
    base_url    : Ollama server URL
    concurrency : max in-flight HTTP requests (semaphore limit)
    batch_size  : texts per /api/embed request (only used when the
                  batch endpoint is available)
    timeout     : per-request timeout in seconds
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        concurrency: int = 8,
        batch_size: int = 16,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._dim: int | None = None
        # Set lazily after first probe
        self._use_batch_endpoint: bool | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _probe_batch_endpoint(self, client: httpx.AsyncClient, sample: str) -> bool:
        """Check if /api/embed (batch) endpoint is available."""
        try:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": [sample]},
                timeout=15.0,
            )
            return resp.status_code == 200 and "embeddings" in resp.json()
        except Exception:
            return False

    async def _embed_one_legacy(self, client: httpx.AsyncClient, text: str) -> List[float]:
        """Single-text embed via /api/embeddings (legacy endpoint)."""
        async with self._semaphore:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    return resp.json()["embedding"]
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def _embed_batch_new(self, client: httpx.AsyncClient, texts: List[str]) -> List[List[float]]:
        """Embed a batch via /api/embed (newer Ollama endpoint)."""
        async with self._semaphore:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": texts},
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    return resp.json()["embeddings"]
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def aembed(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed *texts* asynchronously.

        Parameters
        ----------
        texts    : raw strings (no prefix applied yet)
        is_query : if True, applies model-specific query prefix
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        # Apply query prefix if needed
        formatted = (
            [_format_query(t, self.model) for t in texts]
            if is_query
            else [_format_doc(t, self.model) for t in texts]
        )

        async with httpx.AsyncClient() as client:
            # Probe endpoint once per embedder instance
            if self._use_batch_endpoint is None:
                self._use_batch_endpoint = await self._probe_batch_endpoint(client, formatted[0])

            if self._use_batch_endpoint:
                # Split into batches, fire concurrently (each guarded by semaphore)
                batches = [
                    formatted[i : i + self.batch_size]
                    for i in range(0, len(formatted), self.batch_size)
                ]
                batch_results = await asyncio.gather(
                    *[self._embed_batch_new(client, b) for b in batches]
                )
                all_vectors: List[List[float]] = []
                for br in batch_results:
                    all_vectors.extend(br)
            else:
                # Legacy: one coroutine per text, all concurrent (semaphore limits concurrency)
                all_vectors = await asyncio.gather(
                    *[self._embed_one_legacy(client, t) for t in formatted]
                )

        arr = np.array(all_vectors, dtype=np.float32)
        self._dim = arr.shape[1]
        return arr

    async def aembed_query(self, text: str) -> np.ndarray:
        """Embed a single query string with the appropriate prefix."""
        result = await self.aembed([text], is_query=True)
        return result[0]

    # ------------------------------------------------------------------
    # Sync wrappers (convenience — used only in BM25/non-async paths)
    # ------------------------------------------------------------------

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        return asyncio.run(self.aembed(texts, is_query=is_query))

    def embed_query(self, text: str) -> np.ndarray:
        return asyncio.run(self.aembed_query(text))

    @property
    def dim(self) -> int | None:
        return self._dim
