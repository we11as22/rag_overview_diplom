"""ADK RAG Agent — Wix Knowledge Base Assistant.

Поисковые инструменты (конфиг из 01_retrieval_eval, см. RETRIEVAL_SELECTION.md):
  - search_by_titles  : vector, embeddinggemma, alpha=1.0, top_k=10  (Hit@10=0.74)
  - search_by_chunks  : hybrid linear, embeddinggemma, alpha=0.7, top_k=8  (MRR@10=0.58)
  - open_article      : полный текст → PostgreSQL workspace

Инструменты workspace (PostgreSQL, построчно, с fulltext-поиском):
  - workspace_list    : список всего сохранённого в сессии
  - workspace_read    : чтение по строкам с offset/limit (как Read в IDE)
  - workspace_search  : fulltext поиск с контекстом строк (как grep -n -C)

Большие ответы любых тулов (> TOOL_PERSIST_THRESHOLD_CHARS) автоматически
сохраняются в workspace (after_tool_callback), в контекст — preview + key.

Память: preload_memory_tool + load_memory_tool (PostgresMemoryService).
История: LLM-саммари при переполнении контекста (context_compressor).
"""
from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.tools.tool_context import ToolContext

from tool_callbacks import after_tool_persist, before_model_prepare
from workspace_storage import get_pool, store_text

load_dotenv(override=False)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("LLM_API_KEY", "")
if not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = os.environ.get("LLM_API_BASE", "")

_RAG_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8001")
_ALPHA_TITLES = 1.0   # vector_title, embeddinggemma — Hit@10=0.743
_ALPHA_CHUNKS = 0.7   # hybrid_linear, embeddinggemma — MRR@10=0.577

_model = os.environ["LLM_MODEL"]
_litellm_model = _model if _model.startswith("openai/") else f"openai/{_model}"

_PREVIEW_LINES = int(os.getenv("OPEN_ARTICLE_PREVIEW_LINES", "80"))
_WS_PAGE = int(os.getenv("WORKSPACE_READ_PAGE_LINES", "80"))


def _sid(tool_context: ToolContext) -> str:
    return tool_context._invocation_context.session.id


async def _post(path: str, body: dict) -> list | dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{_RAG_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()


async def search_by_titles(query: str, top_k: int = 10) -> str:
    """Гибридный поиск по заголовкам статей Wix Help Center (vector, alpha=1.0).

    Возвращает топ-10 статей: id, title и наиболее релевантный запросу чанк.
    Используй как первый шаг для любого нового вопроса.

    Args:
        query: Вопрос или поисковый запрос пользователя.
        top_k: Количество статей (1-10, по умолчанию 10).

    Returns:
        JSON-список: id, title, best_chunk, score.
    """
    top_k = max(1, min(top_k, 10))
    results = await _post("/search/hybrid", {
        "query": query, "top_k": top_k, "alpha": _ALPHA_TITLES,
    })
    return json.dumps([
        {"id": r["id"], "title": r["title"],
         "best_chunk": r["best_chunk"], "score": round(r["score"], 4)}
        for r in results
    ], ensure_ascii=False, indent=2)


async def search_by_chunks(query: str, top_k: int = 8) -> str:
    """Гибридный поиск по содержанию документов (hybrid linear, alpha=0.7).

    Ищет конкретные фрагменты текста внутри статей.
    Используй когда нужны точные шаги, настройки или детали.

    Args:
        query: Конкретный вопрос или аспект для поиска.
        top_k: Количество чанков (1-15, по умолчанию 8; для max recall — 15).

    Returns:
        JSON-список: doc_id, title, chunk_text, score.
    """
    top_k = max(1, min(top_k, 15))
    results = await _post("/search/chunks", {
        "query": query, "top_k": top_k, "alpha": _ALPHA_CHUNKS,
    })
    return json.dumps([
        {"doc_id": r["doc_id"], "title": r["title"],
         "chunk_text": r["chunk_text"], "score": round(r["score"], 4)}
        for r in results
    ], ensure_ascii=False, indent=2)


async def open_article(article_id: str, tool_context: ToolContext) -> str:
    """Открыть полный текст статьи. Сохраняет её в workspace для дальнейшего чтения.

    Возвращает первые 80 строк сразу. Если статья длиннее — в ответе будет
    поле "note" с подсказкой как прочитать остаток через workspace_read.

    Args:
        article_id: ID статьи из результатов search_by_titles.

    Returns:
        JSON: title, url, workspace_key, total_lines, preview (первые 80 строк),
              и note если статья обрезана.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{_RAG_URL}/search/article/{article_id}")
        if r.status_code == 404:
            return json.dumps({"error": f"Статья {article_id} не найдена"})
        r.raise_for_status()
        data = r.json()

    full_text = data.get("contents", "")
    title = data.get("title", "")
    url = data.get("url", "")
    ws_key = f"article_{article_id}"
    session = _sid(tool_context)

    total_lines = await store_text(session, ws_key, "open_article", full_text)
    lines = full_text.splitlines() or ([full_text] if full_text else [])
    preview = "\n".join(lines[:_PREVIEW_LINES])

    result = {
        "title": title,
        "url": url,
        "workspace_key": ws_key,
        "total_lines": total_lines,
        "showing_lines": f"1-{min(_PREVIEW_LINES, total_lines)}",
        "preview": preview,
    }
    if total_lines > _PREVIEW_LINES:
        result["note"] = (
            f"Статья содержит {total_lines} строк, показаны первые {_PREVIEW_LINES}. "
            f"Используй workspace_read('{ws_key}', {_PREVIEW_LINES + 1}) "
            f"или workspace_search(query, key='{ws_key}')."
        )
    return json.dumps(result, ensure_ascii=False, indent=2)


async def workspace_list(tool_context: ToolContext) -> str:
    """Показать все записи сохранённые в workspace текущей сессии.

    Returns:
        JSON-список: key, total_lines, created_at для каждой записи.
    """
    p = await get_pool()
    async with p.acquire() as conn:
        rows = await conn.fetch(
            """SELECT key, COUNT(*) AS total_lines, MIN(created_at) AS created_at
               FROM agent_workspace WHERE session_id=$1
               GROUP BY key ORDER BY MIN(created_at)""",
            _sid(tool_context),
        )
    return json.dumps(
        [{"key": r["key"], "total_lines": r["total_lines"],
          "created_at": r["created_at"].isoformat()} for r in rows],
        ensure_ascii=False,
    )


async def workspace_read(
    key: str,
    tool_context: ToolContext,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Читать строки из записи workspace — как чтение файла с позиции.

    Args:
        key:        Имя записи (из workspace_list или workspace_key).
        start_line: Первая строка 1-based (по умолчанию 1).
        end_line:   Последняя строка включительно (по умолчанию start + 79).

    Returns:
        JSON: key, start_line, end_line, total_lines, has_more, lines.
    """
    session = _sid(tool_context)
    if end_line is None:
        end_line = start_line + _WS_PAGE - 1

    p = await get_pool()
    async with p.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_workspace WHERE session_id=$1 AND key=$2",
            session, key,
        )
        if not total:
            return json.dumps({
                "error": f"Запись '{key}' не найдена. Используй workspace_list."
            })
        rows = await conn.fetch(
            """SELECT line_number, line_text FROM agent_workspace
               WHERE session_id=$1 AND key=$2 AND line_number BETWEEN $3 AND $4
               ORDER BY line_number""",
            session, key, start_line, end_line,
        )

    return json.dumps({
        "key": key,
        "start_line": start_line,
        "end_line": min(end_line, total),
        "total_lines": total,
        "has_more": end_line < total,
        "lines": [{"n": r["line_number"], "text": r["line_text"]} for r in rows],
    }, ensure_ascii=False)


async def workspace_search(
    pattern: str,
    tool_context: ToolContext,
    key: str | None = None,
    context_lines: int = 2,
) -> str:
    """Fulltext поиск по workspace сессии — как grep -n -C.

    Args:
        pattern:       Поисковый запрос (слова или фраза).
        key:           Искать только в этой записи (если не задан — по всем).
        context_lines: Строк контекста вокруг совпадения (0-5, по умолчанию 2).

    Returns:
        JSON: total совпадений, matches — список {key, line_number, context}.
    """
    session = _sid(tool_context)
    ctx = max(0, min(context_lines, 5))
    p = await get_pool()

    async with p.acquire() as conn:
        if key:
            hits = await conn.fetch(
                """SELECT key, line_number FROM agent_workspace
                   WHERE session_id=$1 AND key=$2
                     AND to_tsvector('english', line_text) @@ plainto_tsquery('english', $3)
                   ORDER BY line_number LIMIT 30""",
                session, key, pattern,
            )
        else:
            hits = await conn.fetch(
                """SELECT key, line_number FROM agent_workspace
                   WHERE session_id=$1
                     AND to_tsvector('english', line_text) @@ plainto_tsquery('english', $2)
                   ORDER BY key, line_number LIMIT 30""",
                session, pattern,
            )

        if not hits:
            return json.dumps({"pattern": pattern, "total": 0, "matches": []})

        results = []
        for hit in hits:
            ctx_rows = await conn.fetch(
                """SELECT line_number, line_text FROM agent_workspace
                   WHERE session_id=$1 AND key=$2
                     AND line_number BETWEEN $3 AND $4
                   ORDER BY line_number""",
                session, hit["key"],
                max(1, hit["line_number"] - ctx),
                hit["line_number"] + ctx,
            )
            context_str = "\n".join(
                f"{'>>>' if r['line_number'] == hit['line_number'] else '   '}"
                f" {r['line_number']:>4}: {r['line_text']}"
                for r in ctx_rows
            )
            results.append({
                "key": hit["key"],
                "line_number": hit["line_number"],
                "context": context_str,
            })

    return json.dumps({"pattern": pattern, "total": len(results), "matches": results},
                      ensure_ascii=False, indent=2)


model = LiteLlm(model=_litellm_model)

root_agent = Agent(
    name="wix_rag_agent",
    model=model,
    description="Ассистент поддержки Wix с доступом к базе знаний Wix Help Center.",
    before_model_callback=before_model_prepare,
    after_tool_callback=after_tool_persist,
    instruction="""Ты — Wix Assistant, умный помощник по продуктам и настройке Wix.

Ты ведёшь живой диалог: переспрашиваешь когда неясно, помнишь контекст разговора,
отвечаешь кратко если вопрос простой и подробно если сложный.
Не начинай каждый ответ с поиска — сначала пойми что человек хочет.

## Когда искать

Ищи когда: новая тема, нужны конкретные шаги/настройки, пользователь просит найти документацию.
Не ищи когда: уточнение предыдущего ответа, информация уже в workspace этой сессии.

## Инструменты поиска

**search_by_titles(query, top_k=10)** — первый шаг для нового вопроса (α=1.0).
**search_by_chunks(query, top_k=8)** — точные фрагменты и шаги (α=0.7).
**open_article(article_id)** — полный текст статьи в workspace.

Если ответ тула содержит `"status": "persisted"` и `workspace_key` — полный результат
уже в workspace; читай через workspace_read / workspace_search, не проси повторный поиск.

## Инструменты workspace (PostgreSQL, текущая сессия)

**workspace_list()** — что сохранено.
**workspace_read(key, start_line, end_line)** — построчное чтение.
**workspace_search(pattern, key=None)** — fulltext с контекстом строк.

## Как отвечать

- Короткий вопрос → короткий ответ (2-5 предложений)
- Пошаговая инструкция → нумерованный список
- Указывай источник: *Источник: [название статьи]*
- Если не уверен — скажи об этом, не придумывай
- Отвечай на языке пользователя
""",
    tools=[
        search_by_titles,
        search_by_chunks,
        open_article,
        workspace_list,
        workspace_read,
        workspace_search,
        load_memory_tool,
        preload_memory_tool,
    ],
)
