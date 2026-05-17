"""Search router — hybrid search over documents and chunks."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from embedder import embed_query

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    alpha: float = 0.6


class DocResult(BaseModel):
    id: str
    title: str
    best_chunk: str   # наиболее релевантный запросу чанк
    score: float


class ChunkResult(BaseModel):
    doc_id: str
    title: str
    chunk_text: str
    score: float


def _vec_literal(vec: list) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


class ArticleResult(BaseModel):
    id: str
    title: str
    contents: str
    url: str | None = None


@router.get("/article/{article_id}", response_model=ArticleResult)
async def get_article(article_id: str, session: AsyncSession = Depends(get_session)):
    """Get full article text by ID."""
    from db import Document
    doc = await session.get(Document, article_id)
    if doc is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Article {article_id} not found")
    return ArticleResult(id=doc.id, title=doc.title, contents=doc.contents, url=doc.url)


# ---------------------------------------------------------------------------
# Hybrid document search — returns top-K docs with most relevant chunk
# ---------------------------------------------------------------------------

@router.post("/hybrid", response_model=List[DocResult])
async def hybrid_search(req: SearchRequest, session: AsyncSession = Depends(get_session)):
    """Hybrid title search (BM25 + vector).

    Каждый результат содержит id, title и best_chunk —
    наиболее релевантный запросу чанк этого документа.
    """
    q_vec = await embed_query(req.query)
    vec_lit = _vec_literal(q_vec)

    sql = text(f"""
        WITH vec AS (
            SELECT id,
                   1 - (title_vec <=> '{vec_lit}'::vector) AS vec_score
            FROM documents
            WHERE title_vec IS NOT NULL
            ORDER BY title_vec <=> '{vec_lit}'::vector
            LIMIT :limit
        ),
        bm25 AS (
            SELECT id,
                   ts_rank_cd(fts, plainto_tsquery('english', :query), 32) AS bm25_score
            FROM documents
            WHERE fts @@ plainto_tsquery('english', :query)
            LIMIT :limit
        ),
        combined AS (
            SELECT COALESCE(v.id, b.id) AS id,
                   COALESCE(v.vec_score, 0)  AS vec_score,
                   COALESCE(b.bm25_score, 0) AS bm25_score
            FROM vec v
            FULL OUTER JOIN bm25 b ON v.id = b.id
        ),
        normalised AS (
            SELECT id,
                   vec_score  / NULLIF(MAX(vec_score)  OVER (), 0) AS vec_n,
                   bm25_score / NULLIF(MAX(bm25_score) OVER (), 0) AS bm25_n
            FROM combined
        ),
        ranked_docs AS (
            SELECT d.id, d.title,
                   (:alpha * COALESCE(n.vec_n, 0) + (1 - :alpha) * COALESCE(n.bm25_n, 0)) AS score
            FROM normalised n
            JOIN documents d ON d.id = n.id
            ORDER BY score DESC
            LIMIT :top_k
        ),
        best_chunks AS (
            -- Для каждого найденного документа берём чанк с минимальным векторным расстоянием к запросу
            SELECT DISTINCT ON (c.doc_id)
                   c.doc_id,
                   c.chunk_text
            FROM chunks c
            WHERE c.doc_id IN (SELECT id FROM ranked_docs)
              AND c.chunk_vec IS NOT NULL
            ORDER BY c.doc_id, c.chunk_vec <=> '{vec_lit}'::vector
        )
        SELECT r.id, r.title, r.score,
               COALESCE(bc.chunk_text, '') AS best_chunk
        FROM ranked_docs r
        LEFT JOIN best_chunks bc ON bc.doc_id = r.id
        ORDER BY r.score DESC
    """)

    rows = await session.execute(sql, {
        "query": req.query,
        "alpha": req.alpha,
        "limit": req.top_k * 5,
        "top_k": req.top_k,
    })

    return [
        DocResult(id=r.id, title=r.title, best_chunk=r.best_chunk, score=float(r.score))
        for r in rows.mappings()
    ]


# ---------------------------------------------------------------------------
# Hybrid chunk search (content-level)
# ---------------------------------------------------------------------------

@router.post("/chunks", response_model=List[ChunkResult])
async def chunk_search(req: SearchRequest, session: AsyncSession = Depends(get_session)):
    """Hybrid search over document chunks (BM25 + vector)."""
    q_vec = await embed_query(req.query)
    vec_lit = _vec_literal(q_vec)

    sql = text(f"""
        WITH vec AS (
            SELECT c.id AS chunk_id, c.doc_id,
                   1 - (c.chunk_vec <=> '{vec_lit}'::vector) AS vec_score
            FROM chunks c
            WHERE c.chunk_vec IS NOT NULL
            ORDER BY c.chunk_vec <=> '{vec_lit}'::vector
            LIMIT :limit
        ),
        bm25 AS (
            SELECT c.id AS chunk_id, c.doc_id,
                   ts_rank_cd(c.fts, plainto_tsquery('english', :query), 32) AS bm25_score
            FROM chunks c
            WHERE c.fts @@ plainto_tsquery('english', :query)
            LIMIT :limit
        ),
        combined AS (
            SELECT COALESCE(v.chunk_id, b.chunk_id) AS chunk_id,
                   COALESCE(v.doc_id,   b.doc_id)   AS doc_id,
                   COALESCE(v.vec_score, 0)          AS vec_score,
                   COALESCE(b.bm25_score, 0)         AS bm25_score
            FROM vec v
            FULL OUTER JOIN bm25 b ON v.chunk_id = b.chunk_id
        ),
        normalised AS (
            SELECT chunk_id, doc_id,
                   vec_score  / NULLIF(MAX(vec_score)  OVER (), 0) AS vec_n,
                   bm25_score / NULLIF(MAX(bm25_score) OVER (), 0) AS bm25_n
            FROM combined
        )
        SELECT d.title, c.chunk_text, n.doc_id,
               (:alpha * COALESCE(n.vec_n, 0) + (1 - :alpha) * COALESCE(n.bm25_n, 0)) AS score
        FROM normalised n
        JOIN chunks c    ON c.id  = n.chunk_id
        JOIN documents d ON d.id  = n.doc_id
        ORDER BY score DESC
        LIMIT :top_k
    """)

    rows = await session.execute(sql, {
        "query": req.query,
        "alpha": req.alpha,
        "limit": req.top_k * 5,
        "top_k": req.top_k,
    })

    return [
        ChunkResult(doc_id=r.doc_id, title=r.title, chunk_text=r.chunk_text, score=float(r.score))
        for r in rows.mappings()
    ]
