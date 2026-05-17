"""ADK services.py — регистрирует кастомные сервисы.

ADK автоматически загружает этот файл при старте adk web.
- pgmemory:// → PostgresMemoryService
- pgclean://  → CleaningSessionService (при удалении сессии чистит workspace и сохраняет память)
"""
from __future__ import annotations

import asyncio
import logging
import os

from google.adk.cli.service_registry import get_service_registry
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from memory_service import PostgresMemoryService

logger = logging.getLogger(__name__)

_ASYNCPG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
).replace("postgresql+asyncpg://", "postgresql://")


# ---------------------------------------------------------------------------
# Memory service
# ---------------------------------------------------------------------------

def _postgres_memory_factory(uri: str, **kwargs):
    return PostgresMemoryService()

get_service_registry().register_memory_service("pgmemory", _postgres_memory_factory)


# ---------------------------------------------------------------------------
# Session service — оборачивает InMemory, при delete чистит workspace
# ---------------------------------------------------------------------------

class CleaningSessionService(InMemorySessionService):
    """InMemorySessionService + при удалении сессии:
    1. Сохраняет диалог в agent_memory (PostgreSQL)
    2. Удаляет workspace этой сессии из agent_workspace
    """

    def __init__(self, memory_service: PostgresMemoryService):
        super().__init__()
        self._memory_service = memory_service
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(_ASYNCPG_DSN, min_size=1, max_size=5)
        return self._pool

    async def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        # 1. Достаём сессию ДО удаления чтобы сохранить диалог в память
        try:
            session = await self.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            if session:
                await self._memory_service.add_session_to_memory(session)
                logger.info(
                    "Saved session %s to memory before deletion", session_id
                )
        except Exception as e:
            logger.warning("Failed to save memory before delete_session: %s", e)

        # 2. Удаляем workspace этой сессии
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM agent_workspace WHERE session_id=$1",
                    session_id,
                )
            logger.info("Cleaned workspace for session %s", session_id)
        except Exception as e:
            logger.warning("Failed to clean workspace for session %s: %s", session_id, e)

        # 3. Удаляем саму сессию
        await super().delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )


def _cleaning_session_factory(uri: str, **kwargs):
    memory_svc = PostgresMemoryService()
    return CleaningSessionService(memory_service=memory_svc)


get_service_registry().register_session_service("pgclean", _cleaning_session_factory)
