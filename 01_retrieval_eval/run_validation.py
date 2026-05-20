#!/usr/bin/env python3
"""Полный цикл: chunk eval → title eval → pick_top_k.

Usage:
    python run_validation.py
    python run_validation.py --models embeddinggemma
    python run_validation.py --skip-rebuild   # если индексы уже на отфильтрованном корпусе
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("\n" + "=" * 60)
    print(">", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate retrieval + pick top_k")
    p.add_argument("--models", help="Override OLLAMA_EMBED_MODELS (comma-separated)")
    p.add_argument("--skip-rebuild", action="store_true", help="Не пересобирать Chroma")
    p.add_argument("--dry-run", action="store_true", help="5 запросов для smoke test")
    args = p.parse_args()

    import os
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    if args.models:
        env["OLLAMA_EMBED_MODELS"] = args.models

    chunk_cmd = [sys.executable, "run_eval.py", "--skip-chain"]
    if not args.skip_rebuild:
        chunk_cmd.append("--force-rebuild")
    if args.dry_run:
        chunk_cmd.append("--dry-run")

    title_cmd = [sys.executable, "eval_title_search.py"]
    if args.dry_run:
        title_cmd.append("--dry-run")
    if args.models:
        title_cmd.extend(["--models", args.models])

    _run(chunk_cmd, env=env)
    _run(title_cmd, env=env)
    _run([sys.executable, "pick_top_k.py"], env=env)

    print("\nГотово:")
    print(f"  {ROOT / 'results' / 'report.md'}")
    print(f"  {ROOT / 'results' / 'title_search_report.md'}")
    print(f"  {ROOT / 'results' / 'pick_top_k.md'}  ← top_k для агента")


if __name__ == "__main__":
    main()
