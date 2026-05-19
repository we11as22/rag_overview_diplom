#!/usr/bin/env python3
"""Trigger corpus ingestion into rag_service.

Usage:
    python ingest.py
    python ingest.py --clear
    python ingest.py --path /data/corpus.jsonl   # rag_service в Docker
    python ingest.py --path ../data/corpus.jsonl   # rag_service локально (start.sh)
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOCAL_CORPUS = _ROOT.parent / "data" / "corpus.jsonl"
_DEFAULT_DOCKER_CORPUS = "/data/corpus.jsonl"
RAG_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8001").rstrip("/")


async def main(clear: bool, corpus_path: str) -> None:
    payload = {"corpus_path": corpus_path, "clear_existing": clear}
    print(f"RAG: {RAG_URL}")
    print(f"Starting ingestion from: {corpus_path} (clear={clear})...")
    async with httpx.AsyncClient(timeout=3600.0) as client:
        resp = await client.post(f"{RAG_URL}/ingest", json=payload)
        resp.raise_for_status()
        print(resp.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "Путь к corpus.jsonl на стороне rag_service "
            f"(локально по умолчанию: {_DEFAULT_LOCAL_CORPUS}; "
            f"в Docker rag: {_DEFAULT_DOCKER_CORPUS})"
        ),
    )
    parser.add_argument(
        "--docker-rag",
        action="store_true",
        help=f"Корпус внутри контейнера rag_service ({_DEFAULT_DOCKER_CORPUS})",
    )
    args = parser.parse_args()

    if args.path:
        corpus = args.path
    elif args.docker_rag:
        corpus = _DEFAULT_DOCKER_CORPUS
    else:
        corpus = str(_DEFAULT_LOCAL_CORPUS)

    asyncio.run(main(args.clear, corpus))
