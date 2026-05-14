"""Dense vector retriever backed by ChromaDB with async indexing."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List, Tuple

import chromadb

from embeddings import OllamaEmbedder


def _safe_collection_name(model_name: str) -> str:
    """ChromaDB collection names must be [3-63 chars, alphanumeric + hyphens]."""
    name = re.sub(r"[^a-zA-Z0-9-]", "-", model_name)
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) < 3:
        name = (name + "---")[:3]
    return name[:63]


# ~4000 chars ≈ ~1000 tokens — safe for all three models
_MAX_DOC_CHARS = 4000


def _truncate(text: str, max_chars: int = _MAX_DOC_CHARS) -> str:
    return text[:max_chars] if len(text) > max_chars else text


class VectorRetriever:
    """Dense retriever: embeds corpus with Ollama (async) and stores in ChromaDB."""

    def __init__(
        self,
        corpus: List[dict],
        embedder: OllamaEmbedder,
        persist_dir: str | Path,
        force_rebuild: bool = False,
    ) -> None:
        self._corpus = corpus
        self._embedder = embedder
        self._persist_dir = Path(persist_dir)
        self._collection_name = _safe_collection_name(embedder.model)

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        collections = self._client.list_collections()
        existing_names = {c.name if hasattr(c, "name") else c for c in collections}
        already_exists = self._collection_name in existing_names

        if force_rebuild and already_exists:
            self._client.delete_collection(self._collection_name)
            already_exists = False

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Deferred async build — caller must await ensure_built() before use.
        # Also rebuild if collection exists but is empty (interrupted previous build).
        populated = already_exists and self._collection.count() > 0
        self._needs_build = not populated

    async def ensure_built(self) -> None:
        """Async index build if needed. Must be awaited before first use."""
        if self._needs_build:
            await self._build_index_async()
            self._needs_build = False

    async def _build_index_async(self) -> None:
        """Embed all corpus documents asynchronously and upsert into ChromaDB."""
        print(f"  Building vector index for '{self._embedder.model}' ({len(self._corpus)} docs)...")

        batch_size = 64
        for i in range(0, len(self._corpus), batch_size):
            batch = self._corpus[i : i + batch_size]
            texts = [_truncate(f"{doc.get('title', '')} {doc.get('contents', '')}") for doc in batch]
            ids = [doc["id"] for doc in batch]

            end = min(i + batch_size, len(self._corpus))
            print(f"    Embedding {i + 1}-{end} / {len(self._corpus)}", end="\r")

            vectors = await self._embedder.aembed(texts, is_query=False)

            self._collection.upsert(
                ids=ids,
                embeddings=vectors.tolist(),
                metadatas=[{"title": doc.get("title", ""), "id": doc["id"]} for doc in batch],
            )

        print(f"\n  Index built: {self._collection.count()} documents stored.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Sync retrieve — creates its own event loop, safe outside async context."""
        query_vec = asyncio.run(self._embedder.aembed_query(query))
        return self._retrieve_with_vec(query_vec, top_k)

    async def aretrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Async retrieve — use from async evaluation loops."""
        query_vec = await self._embedder.aembed_query(query)
        return self._retrieve_with_vec(query_vec, top_k)

    def _retrieve_with_vec(self, query_vec, top_k: int) -> List[Tuple[str, float]]:
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["distances"],
        )
        doc_ids = results["ids"][0]
        scores = [1.0 - float(d) for d in results["distances"][0]]
        return list(zip(doc_ids, scores))
