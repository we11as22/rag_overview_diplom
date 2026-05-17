"""Ingest router — loads WixQA corpus into PostgreSQL + pgvector."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session, Document, Chunk
from embedder import embed_texts

router = APIRouter(tags=["ingest"])

CHUNK_TARGET = 600    # целевой размер чанка в символах
CHUNK_OVERLAP_SENTS = 1  # перекрытие: последнее предложение предыдущего чанка
BATCH = 32


def _split_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по точке/восклику/вопросу."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _chunk(text: str) -> List[str]:
    """Параграфный чанкинг с объединением коротких параграфов.

    1. Разбить по \\n на параграфы
    2. Объединять параграфы пока < CHUNK_TARGET символов
    3. Параграф > CHUNK_TARGET — разбить по предложениям
    4. Перекрытие: последнее предложение предыдущего чанка
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0
    last_sentence: str = ""

    for para in paragraphs:
        # Параграф помещается в текущий чанк (или чанк пустой)
        if current_len + len(para) + 1 <= CHUNK_TARGET or not current_parts:
            current_parts.append(para)
            current_len += len(para) + 1
        else:
            # Сохраняем накопленный чанк
            chunk_text = " ".join(current_parts)
            # Добавляем перекрытие из предыдущего чанка
            if last_sentence:
                chunk_text = last_sentence + " " + chunk_text
            chunks.append(chunk_text.strip())

            # Запоминаем последнее предложение для следующего перекрытия
            sents = _split_sentences(" ".join(current_parts))
            last_sentence = sents[-1] if sents else ""

            # Начинаем новый чанк
            current_parts = [para]
            current_len = len(para)

        # Если параграф сам по себе длиннее target — разбиваем по предложениям
        if len(para) > CHUNK_TARGET and len(current_parts) == 1:
            sents = _split_sentences(para)
            if len(sents) > 1:
                current_parts = []
                current_len = 0
                buf: List[str] = []
                buf_len = 0
                for sent in sents:
                    if buf_len + len(sent) > CHUNK_TARGET and buf:
                        chunk_text = " ".join(buf)
                        if last_sentence:
                            chunk_text = last_sentence + " " + chunk_text
                        chunks.append(chunk_text.strip())
                        last_sentence = buf[-1]
                        buf = [sent]
                        buf_len = len(sent)
                    else:
                        buf.append(sent)
                        buf_len += len(sent) + 1
                if buf:
                    current_parts = buf
                    current_len = buf_len

    # Последний чанк
    if current_parts:
        chunk_text = " ".join(current_parts)
        if last_sentence:
            chunk_text = last_sentence + " " + chunk_text
        chunks.append(chunk_text.strip())

    return [c for c in chunks if c]


class IngestRequest(BaseModel):
    corpus_path: str = "/data/corpus.jsonl"
    clear_existing: bool = False


@router.post("")
async def ingest(req: IngestRequest, session: AsyncSession = Depends(get_session)):
    corpus_path = Path(req.corpus_path)
    if not corpus_path.exists():
        raise HTTPException(404, f"File not found: {corpus_path}")

    if req.clear_existing:
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        await session.commit()

    # Check already ingested
    result = await session.execute(text("SELECT COUNT(*) FROM documents"))
    existing = result.scalar()

    docs = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]

    if existing >= len(docs):
        return {"status": "already_ingested", "docs": existing}

    ingested = 0
    for i in range(0, len(docs), BATCH):
        batch = docs[i: i + BATCH]

        # --- Documents — коммитим ПЕРВЫМИ, до чанков (foreign key) ---
        titles = [d.get("title", "") for d in batch]
        title_vecs = await embed_texts(titles, is_query=False)

        for doc, tvec in zip(batch, title_vecs):
            existing_doc = await session.get(Document, doc["id"])
            if existing_doc:
                continue
            session.add(Document(
                id=doc["id"],
                url=doc.get("url", ""),
                title=doc.get("title", ""),
                contents=doc.get("contents", ""),
                article_type=doc.get("article_type", ""),
                title_vec=tvec,
            ))

        await session.commit()  # documents в БД — теперь можно вставлять чанки

        # --- Chunks ---
        all_chunks: List[tuple[str, int, str]] = []
        for doc in batch:
            for ci, chunk_text in enumerate(_chunk(doc.get("contents", ""))):
                all_chunks.append((doc["id"], ci, chunk_text))

        chunk_texts = [c[2] for c in all_chunks]
        for j in range(0, len(chunk_texts), BATCH):
            sub = chunk_texts[j: j + BATCH]
            vecs = await embed_texts(sub, is_query=False)
            for (doc_id, ci, ct), vec in zip(all_chunks[j: j + BATCH], vecs):
                session.add(Chunk(doc_id=doc_id, chunk_index=ci, chunk_text=ct, chunk_vec=vec))

        await session.commit()
        ingested += len(batch)
        print(f"Ingested {ingested}/{len(docs)} docs", flush=True)

    return {"status": "ok", "ingested": ingested}
