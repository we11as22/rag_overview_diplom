"""PostgreSQL workspace — построчное хранение больших результатов тулов.

Аналог Hermes maybe_persist_tool_result, но в agent_workspace вместо файлов sandbox.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Optional

_ASYNCPG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
).replace("postgresql+asyncpg://", "postgresql://")

PERSIST_THRESHOLD_CHARS = int(os.getenv("TOOL_PERSIST_THRESHOLD_CHARS", "8000"))
PREVIEW_CHARS = int(os.getenv("TOOL_PERSIST_PREVIEW_CHARS", "2000"))

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        import asyncpg

        _pool = await asyncpg.create_pool(_ASYNCPG_DSN, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM agent_workspace WHERE created_at < NOW() - INTERVAL '7 days'"
            )
    return _pool


def generate_preview(content: str, max_chars: int = PREVIEW_CHARS) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[: last_nl + 1]
    return truncated, True


async def store_lines(
    session_id: str,
    key: str,
    tool_name: str,
    lines: list[str],
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM agent_workspace WHERE session_id=$1 AND key=$2",
            session_id,
            key,
        )
        if lines:
            await conn.executemany(
                "INSERT INTO agent_workspace(session_id,key,tool_name,line_number,line_text)"
                " VALUES($1,$2,$3,$4,$5)",
                [
                    (session_id, key, tool_name, i + 1, ln)
                    for i, ln in enumerate(lines)
                ],
            )
    return len(lines)


async def store_text(
    session_id: str,
    key: str,
    tool_name: str,
    text: str,
) -> int:
    lines = text.splitlines()
    if not lines and text:
        lines = [text]
    return await store_lines(session_id, key, tool_name, lines)


def build_persisted_stub(
    *,
    workspace_key: str,
    tool_name: str,
    original_size: int,
    preview: str,
    has_more: bool,
    total_lines: int,
) -> str:
    size_kb = original_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    payload = {
        "status": "persisted",
        "workspace_key": workspace_key,
        "tool": tool_name,
        "original_size_chars": original_size,
        "total_lines": total_lines,
        "preview": preview,
        "hint": (
            "Полный результат в workspace. Используй workspace_list(), "
            f"workspace_read('{workspace_key}', start_line, end_line) или "
            f"workspace_search(pattern, key='{workspace_key}')."
        ),
    }
    if has_more:
        payload["note"] = "Preview обрезан; читай дальше через workspace_read."
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def maybe_persist(
    session_id: str,
    tool_name: str,
    content: str,
    *,
    key: Optional[str] = None,
    force: bool = False,
) -> str:
    """Если content длинный — в Postgres + stub в контекст (как Hermes layer 2)."""
    if not force and len(content) <= PERSIST_THRESHOLD_CHARS:
        return content

    ws_key = key or f"{tool_name}_{uuid.uuid4().hex[:12]}"
    lines = content.splitlines()
    if not lines and content:
        lines = [content]
    total_lines = await store_lines(session_id, ws_key, tool_name, lines)
    preview, has_more = generate_preview(content, PREVIEW_CHARS)
    return build_persisted_stub(
        workspace_key=ws_key,
        tool_name=tool_name,
        original_size=len(content),
        preview=preview,
        has_more=has_more,
        total_lines=total_lines,
    )


def coerce_tool_response_text(tool_response) -> str:
    if tool_response is None:
        return ""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        if "result" in tool_response and isinstance(tool_response["result"], str):
            return tool_response["result"]
        return json.dumps(tool_response, ensure_ascii=False)
    return str(tool_response)
