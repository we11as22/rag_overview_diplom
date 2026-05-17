"""PostgreSQL-backed memory service for ADK.

Хранит историю разговоров в таблице agent_memory.
Поиск — полнотекстовый через tsvector (те же механизмы что и для RAG).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import asyncpg
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/rag",
)
# asyncpg DSN — без +asyncpg prefix
_ASYNCPG_DSN = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL PRIMARY KEY,
    app_name    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    author      TEXT,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_json TEXT NOT NULL,
    fts         TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', content_json)
                ) STORED
);
CREATE INDEX IF NOT EXISTS agent_memory_user_idx ON agent_memory(app_name, user_id);
CREATE INDEX IF NOT EXISTS agent_memory_fts_idx  ON agent_memory USING GIN(fts);
"""


class PostgresMemoryService(BaseMemoryService):
    """Персистентная память агента в PostgreSQL.

    add_session_to_memory — вызывается ADK после каждой сессии автоматически.
    search_memory          — вызывается load_memory_tool при запросе.
    """

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(_ASYNCPG_DSN, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(_INIT_SQL)
        return self._pool

    async def add_session_to_memory(self, session) -> None:
        """Сохраняет все события сессии в память."""
        pool = await self._get_pool()
        events = getattr(session, "events", []) or []
        if not events:
            return

        rows = []
        for event in events:
            if not event.content or not event.content.parts:
                continue
            text_parts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if not text_parts:
                continue
            content_json = json.dumps({
                "role": event.author,
                "text": " ".join(text_parts),
            })
            rows.append((
                session.app_name,
                session.user_id,
                session.id,
                event.author,
                datetime.fromtimestamp(event.timestamp, tz=timezone.utc) if event.timestamp else datetime.now(tz=timezone.utc),
                content_json,
            ))

        if rows:
            async with pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO agent_memory (app_name, user_id, session_id, author, ts, content_json)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                    """,
                    rows,
                )

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        """Полнотекстовый поиск по истории разговоров пользователя."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT author, ts, content_json,
                       ts_rank_cd(fts, plainto_tsquery('english', $3)) AS rank
                FROM agent_memory
                WHERE app_name = $1
                  AND user_id  = $2
                  AND fts @@ plainto_tsquery('english', $3)
                ORDER BY rank DESC, ts DESC
                LIMIT 10
                """,
                app_name, user_id, query,
            )

        memories = []
        for row in rows:
            data = json.loads(row["content_json"])
            memories.append(MemoryEntry(
                content=types.Content(
                    role=data.get("role", "user"),
                    parts=[types.Part(text=data.get("text", ""))],
                ),
                author=row["author"],
                timestamp=row["ts"].isoformat(),
            ))

        return SearchMemoryResponse(memories=memories)
