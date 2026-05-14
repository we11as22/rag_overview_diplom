"""Download WixQA dataset from HuggingFace and save to ../data/."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = (Path(__file__).parent / ".." / "data").resolve()


def download_corpus() -> None:
    """Download wix_kb_corpus and save to ../data/corpus.jsonl."""
    print("Downloading corpus (wix_kb_corpus)...")
    ds = load_dataset("Wix/WixQA", "wix_kb_corpus", trust_remote_code=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "corpus.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split_name, split_data in ds.items():
            for row in tqdm(split_data, desc=f"  split={split_name}"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} documents -> {out_path}")


def download_qa() -> None:
    """Download wixqa_expertwritten QA pairs and save to ../data/qa_expertwritten.jsonl."""
    print("Downloading QA pairs (wixqa_expertwritten)...")
    ds = load_dataset("Wix/WixQA", "wixqa_expertwritten", trust_remote_code=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "qa_expertwritten.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split_name, split_data in ds.items():
            for row in tqdm(split_data, desc=f"  split={split_name}"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} QA pairs -> {out_path}")


if __name__ == "__main__":
    download_corpus()
    download_qa()
    print("\nDone. Files saved to:", DATA_DIR)
