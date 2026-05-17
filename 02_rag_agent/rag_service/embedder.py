"""Ollama embeddings client — async, with query prefix logic."""
from __future__ import annotations

import asyncio
import os
from typing import List

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")


def _format_query(text: str, model: str) -> str:
    m = model.lower()
    if "embeddinggemma" in m or "gemma" in m:
        return f"task: search result | query: {text}"
    if "qwen3" in m or "qwen" in m:
        return f"Instruct: Given a question, retrieve relevant documents that answer the question\nQuery:{text}"
    return text


async def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed a list of texts via Ollama. Returns list of float vectors."""
    formatted = [_format_query(t, EMBED_MODEL) if is_query else t for t in texts]
    results: List[List[float]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Try batch endpoint first
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": formatted},
            )
            if resp.status_code == 200 and "embeddings" in resp.json():
                return resp.json()["embeddings"]
        except Exception:
            pass

        # Legacy: one request per text
        sem = asyncio.Semaphore(8)

        async def _one(text: str) -> List[float]:
            async with sem:
                r = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                    timeout=120.0,
                )
                r.raise_for_status()
                return r.json()["embedding"]

        results = await asyncio.gather(*[_one(t) for t in formatted])

    return list(results)


async def embed_query(text: str) -> List[float]:
    vecs = await embed_texts([text], is_query=True)
    return vecs[0]
