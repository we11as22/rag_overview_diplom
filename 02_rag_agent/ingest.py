#!/usr/bin/env python3
"""Trigger corpus ingestion into rag_service.

Usage:
    python ingest.py
    python ingest.py --clear       # drop existing data first
    python ingest.py --path /custom/path/corpus.jsonl
"""
import argparse
import asyncio
from pathlib import Path

import httpx

RAG_URL = "http://localhost:8001"
# Default: ../data/corpus.jsonl relative to this script (local run)
_DEFAULT_PATH = str((Path(__file__).parent.parent / "data" / "corpus.jsonl").resolve())


async def main(clear: bool, corpus_path: str):
    payload = {"corpus_path": corpus_path, "clear_existing": clear}
    print(f"Starting ingestion from: {corpus_path} (clear={clear})...")
    async with httpx.AsyncClient(timeout=3600.0) as client:
        resp = await client.post(f"{RAG_URL}/ingest", json=payload)
        resp.raise_for_status()
        print(resp.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--path", default=_DEFAULT_PATH,
                        help="Path to corpus.jsonl (default: ../data/corpus.jsonl)")
    args = parser.parse_args()
    asyncio.run(main(args.clear, args.path))
