"""ADK RAG Agent — Wix Knowledge Base Assistant.

Поисковые инструменты (гибридные, alpha=0.6):
  - search_by_titles      : гибридный поиск по заголовкам
  - search_by_chunks      : гибридный поиск по содержанию (чанки)
  - open_article          : сохраняет полный текст в артефакт, возвращает первые N строк

Инструменты артефактов:
  - list_saved_articles   : список сохранённых документов
  - read_article_lines    : чтение артефакта по диапазону строк (как Read в IDE)
  - search_in_article     : полнотекстовый поиск по конкретному артефакту
  - search_in_articles    : полнотекстовый поиск по нескольким артефактам

Память: preload_memory_tool + load_memory_tool (PostgresMemoryService).
Стриминг: автоматически через ADK SSE.
"""
from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv(override=False)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("LLM_API_KEY", "")
if not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = os.environ.get("LLM_API_BASE", "")

_RAG_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8001")
_ALPHA_CHUNKS = 0.6   # лучший по MRR@10: hybrid_linear, embeddinggemma, α=0.6
_ALPHA_TITLES = 1.0   # лучший по Hit@10: pure vector, embeddinggemma

_ARTIFACT_PREVIEW_LINES = 80   # строк показывается агенту сразу при open_article
_ARTIFACT_PAGE_SIZE    = 100   # строк за один вызов read_article_lines

_model = os.environ["LLM_MODEL"]
_litellm_model = _model if _model.startswith("openai/") else f"openai/{_model}"

# Максимум пар сообщений в контексте LLM
_MAX_HISTORY_TURNS = 20


# ---------------------------------------------------------------------------
# History trimming callback
# ---------------------------------------------------------------------------

def _trim_history(callback_context, llm_request) -> None:
    """Обрезает историю до последних _MAX_HISTORY_TURNS пар."""
    contents = llm_request.contents
    if contents and len(contents) > _MAX_HISTORY_TURNS * 2:
        llm_request.contents = contents[-(_MAX_HISTORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _post(path: str, body: dict) -> list | dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{_RAG_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _article_filename(article_id: str) -> str:
    """Имя файла артефакта для статьи."""
    return f"article_{article_id}.txt"


def _lines_preview(text: str, n: int) -> tuple[list[str], int]:
    """Возвращает первые n строк и общее количество строк."""
    lines = text.splitlines()
    return lines[:n], len(lines)


# ---------------------------------------------------------------------------
# Tool 1: поиск по заголовкам
# ---------------------------------------------------------------------------

async def search_by_titles(query: str, top_k: int = 5) -> str:
    """Гибридный поиск по заголовкам документов Wix Help Center (BM25 + vector, alpha=0.6).

    Возвращает список наиболее релевантных статей с кратким содержанием.
    Используй как первый шаг — чтобы найти какие статьи существуют по теме.

    Args:
        query: Поисковый запрос или вопрос пользователя.
        top_k: Сколько статей вернуть (1-10, по умолчанию 5).

    Returns:
        JSON-список: id, title, summary (первые 600 символов), score.
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


# ---------------------------------------------------------------------------
# Tool 2: поиск по чанкам
# ---------------------------------------------------------------------------

async def search_by_chunks(query: str, top_k: int = 6) -> str:
    """Гибридный поиск по содержанию документов (чанки, BM25 + vector, alpha=0.6).

    Ищет конкретные фрагменты текста внутри статей.
    Используй когда нужны точные инструкции, шаги или детали.

    Args:
        query: Конкретный вопрос или аспект для поиска.
        top_k: Сколько чанков вернуть (1-15, по умолчанию 6).

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


def _artifacts_enabled(tool_context: ToolContext) -> bool:
    return tool_context._invocation_context.artifact_service is not None


_NO_ARTIFACTS_MSG = json.dumps({
    "error": "Артефакты не включены. Запусти агента с флагом "
             "--artifact_service_uri \"file://$HOME/.adk_artifacts\""
})


# ---------------------------------------------------------------------------
# Tool 3: открыть статью → сохранить артефакт, показать первые строки
# ---------------------------------------------------------------------------

async def open_article(article_id: str, tool_context: ToolContext) -> str:
    """Открыть статью: сохраняет полный текст в артефакт, возвращает первые строки.

    Полный текст сохраняется в артефакт — к нему можно обращаться через
    read_article_lines и search_in_article без повторного обращения к БД.

    Args:
        article_id: ID статьи из результатов search_by_titles.

    Returns:
        JSON: title, url, total_lines, preview (первые 80 строк),
              artifact_name — имя артефакта для дальнейшего чтения.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{_RAG_URL}/search/article/{article_id}")
        if r.status_code == 404:
            return json.dumps({"error": f"Статья {article_id} не найдена"})
        r.raise_for_status()
        data = r.json()

    full_text = data.get("contents", "")
    title     = data.get("title", "")
    url       = data.get("url", "")

    # Сохраняем полный текст в артефакт (если artifact_service доступен)
    artifact_name = _article_filename(article_id)
    artifacts_available = _artifacts_enabled(tool_context)
    if artifacts_available:
        await tool_context.save_artifact(
            filename=artifact_name,
            artifact=types.Part.from_text(text=full_text),
            custom_metadata={"title": title, "url": url, "article_id": article_id},
        )

    preview_lines, total_lines = _lines_preview(full_text, _ARTIFACT_PREVIEW_LINES)

    result = {
        "title": title,
        "url": url,
        "artifact_name": artifact_name if artifacts_available else None,
        "artifacts_enabled": artifacts_available,
        "total_lines": total_lines,
        "showing_lines": f"1-{len(preview_lines)}",
        "preview": "\n".join(preview_lines),
    }
    if total_lines > _ARTIFACT_PREVIEW_LINES:
        if artifacts_available:
            result["note"] = (
                f"Документ содержит {total_lines} строк. "
                f"Показаны строки 1-{_ARTIFACT_PREVIEW_LINES}. "
                f"Используй read_article_lines('{artifact_name}', ...) "
                f"для чтения остальных строк или search_in_article для поиска."
            )
        else:
            result["note"] = (
                f"Документ содержит {total_lines} строк, показаны первые {_ARTIFACT_PREVIEW_LINES}. "
                f"Артефакты не включены — запусти агента с --artifact_service_uri чтобы читать документы целиком."
            )

    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool 4: список сохранённых артефактов
# ---------------------------------------------------------------------------

async def list_saved_articles(tool_context: ToolContext) -> str:
    """Показать список статей, открытых в этой сессии (сохранённых в артефакты).

    Returns:
        JSON-список: artifact_name для каждой сохранённой статьи.
    """
    if not _artifacts_enabled(tool_context):
        return _NO_ARTIFACTS_MSG
    all_keys = await tool_context.list_artifacts()
    article_keys = [k for k in all_keys if k.startswith("article_")]
    return json.dumps({
        "saved_articles": article_keys,
        "count": len(article_keys),
        "hint": "Используй read_article_lines или search_in_article для работы с ними.",
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool 5: чтение артефакта по строкам
# ---------------------------------------------------------------------------

async def read_article_lines(
    artifact_name: str,
    start_line: int,
    tool_context: ToolContext,
    end_line: int | None = None,
) -> str:
    """Читать строки из сохранённого артефакта — как чтение файла с позиции.

    Args:
        artifact_name: Имя артефакта (из open_article или list_saved_articles).
        start_line: Первая строка (1-based).
        end_line: Последняя строка включительно (если не задана — start_line + 99).

    Returns:
        JSON: artifact_name, start_line, end_line, total_lines, content,
              и has_more если есть ещё строки после end_line.
    """
    if not _artifacts_enabled(tool_context):
        return _NO_ARTIFACTS_MSG
    part = await tool_context.load_artifact(artifact_name)
    if part is None:
        return json.dumps({"error": f"Артефакт '{artifact_name}' не найден. Сначала вызови open_article."})

    lines = part.text.splitlines()
    total = len(lines)
    start = max(1, start_line)
    end   = min(total, end_line if end_line is not None else start + _ARTIFACT_PAGE_SIZE - 1)

    return json.dumps({
        "artifact_name": artifact_name,
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "has_more": end < total,
        "content": "\n".join(lines[start - 1:end]),
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool 6: поиск по одному артефакту
# ---------------------------------------------------------------------------

async def search_in_article(
    artifact_name: str,
    pattern: str,
    tool_context: ToolContext,
    context_lines: int = 2,
) -> str:
    """Полнотекстовый поиск (regex) по конкретному сохранённому документу.

    Args:
        artifact_name: Имя артефакта (из open_article или list_saved_articles).
        pattern: Поисковый паттерн (подстрока или регулярное выражение).
        context_lines: Сколько строк контекста вокруг каждого совпадения (0-5).

    Returns:
        JSON: список совпадений с номерами строк и контекстом.
    """
    if not _artifacts_enabled(tool_context):
        return _NO_ARTIFACTS_MSG
    part = await tool_context.load_artifact(artifact_name)
    if part is None:
        return json.dumps({"error": f"Артефакт '{artifact_name}' не найден. Сначала вызови open_article."})

    lines = part.text.splitlines()
    ctx   = max(0, min(context_lines, 5))
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)

    matches = []
    for i, line in enumerate(lines):
        if rx.search(line):
            start = max(0, i - ctx)
            end   = min(len(lines), i + ctx + 1)
            snippet = []
            for j in range(start, end):
                prefix = ">>>" if j == i else "   "
                snippet.append(f"{prefix} {j+1:>4}: {lines[j]}")
            matches.append({
                "line_number": i + 1,
                "match": line.strip(),
                "context": "\n".join(snippet),
            })
            if len(matches) >= 30:
                break

    return json.dumps({
        "artifact_name": artifact_name,
        "pattern": pattern,
        "total_matches": len(matches),
        "matches": matches,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool 7: поиск по нескольким артефактам
# ---------------------------------------------------------------------------

async def search_in_articles(
    artifact_names: list[str],
    pattern: str,
    tool_context: ToolContext,
) -> str:
    """Полнотекстовый поиск по нескольким сохранённым документам сразу.

    Args:
        artifact_names: Список имён артефактов для поиска.
        pattern: Поисковый паттерн (подстрока или регулярное выражение).

    Returns:
        JSON: словарь {artifact_name: список совпадений с номерами строк}.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)

    results = {}
    for name in artifact_names:
        part = await tool_context.load_artifact(name)
        if part is None:
            results[name] = {"error": "артефакт не найден"}
            continue
        lines = part.text.splitlines()
        hits = [
            {"line_number": i + 1, "text": line.strip()}
            for i, line in enumerate(lines)
            if rx.search(line)
        ][:20]
        results[name] = {"matches": hits, "total": len(hits)}

    return json.dumps({"pattern": pattern, "results": results},
                      ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PostgreSQL Workspace — persist tool outputs, read by lines, search
# Паттерн из Hermes: большие выводы сохраняются построчно,
# агент читает их через offset/limit и ищет через fulltext.
# ---------------------------------------------------------------------------

_ASYNCPG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
).replace("postgresql+asyncpg://", "postgresql://")

_WS_PAGE = 80   # строк за один вызов workspace_read по умолчанию


async def _ws_pool():
    """Lazy asyncpg pool — создаётся один раз при первом вызове."""
    if not hasattr(_ws_pool, "_pool") or _ws_pool._pool is None:
        import asyncpg
        _ws_pool._pool = await asyncpg.create_pool(_ASYNCPG_DSN, min_size=1, max_size=5)
    return _ws_pool._pool

_ws_pool._pool = None


async def workspace_write(key: str, content: str, tool_context: ToolContext) -> str:
    """Сохранить текст в рабочее пространство сессии построчно.

    Используй чтобы сохранить большой вывод инструмента для последующего
    чтения и поиска — как запись в файл. Перезаписывает предыдущую запись с тем же key.

    Args:
        key:     Имя записи (латиницей, без пробелов). Например: "search_results", "article_abc".
        content: Текст для сохранения (любой, в том числе многострочный).

    Returns:
        JSON: key, total_lines, session_id.
    """
    session_id = tool_context._invocation_context.session.id
    lines = content.splitlines()
    pool = await _ws_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM agent_workspace WHERE session_id=$1 AND key=$2",
            session_id, key,
        )
        if lines:
            await conn.executemany(
                """INSERT INTO agent_workspace (session_id, key, tool_name, line_number, line_text)
                   VALUES ($1, $2, $3, $4, $5)""",
                [
                    (session_id, key, "workspace_write", i + 1, line)
                    for i, line in enumerate(lines)
                ],
            )
    return json.dumps({"key": key, "total_lines": len(lines), "session_id": session_id})


async def workspace_list(tool_context: ToolContext) -> str:
    """Показать все записи в рабочем пространстве текущей сессии.

    Returns:
        JSON: список записей с key, total_lines, created_at.
    """
    session_id = tool_context._invocation_context.session.id
    pool = await _ws_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT key,
                      COUNT(*)        AS total_lines,
                      MIN(created_at) AS created_at
               FROM agent_workspace
               WHERE session_id = $1
               GROUP BY key
               ORDER BY MIN(created_at)""",
            session_id,
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
    """Читать строки из записи рабочего пространства — как read_file с offset/limit.

    Args:
        key:        Имя записи (из workspace_list).
        start_line: Первая строка (1-based).
        end_line:   Последняя строка включительно (если не задана — start + 79).

    Returns:
        JSON: key, start_line, end_line, total_lines, has_more, lines (список строк с номерами).
    """
    session_id = tool_context._invocation_context.session.id
    if end_line is None:
        end_line = start_line + _WS_PAGE - 1

    pool = await _ws_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_workspace WHERE session_id=$1 AND key=$2",
            session_id, key,
        )
        if total == 0:
            return json.dumps({"error": f"Запись '{key}' не найдена. Используй workspace_list."})

        rows = await conn.fetch(
            """SELECT line_number, line_text
               FROM agent_workspace
               WHERE session_id=$1 AND key=$2
                 AND line_number BETWEEN $3 AND $4
               ORDER BY line_number""",
            session_id, key, start_line, end_line,
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
    """Полнотекстовый поиск по рабочему пространству сессии.

    Поиск по всем записям или по конкретной. Возвращает номера строк
    и контекст вокруг каждого совпадения — как grep -n -C.

    Args:
        pattern:       Поисковый запрос (слова через пробел) или фраза.
        key:           Искать только в этой записи (если не задан — по всем).
        context_lines: Строк контекста вокруг совпадения (0-5, по умолчанию 2).

    Returns:
        JSON: список совпадений с key, line_number, matched_line, context.
    """
    session_id = tool_context._invocation_context.session.id
    context_lines = max(0, min(context_lines, 5))
    pool = await _ws_pool()

    async with pool.acquire() as conn:
        # Fulltext поиск через tsvector — находим (key, line_number) совпадений
        if key:
            hit_rows = await conn.fetch(
                """SELECT key, line_number
                   FROM agent_workspace
                   WHERE session_id=$1 AND key=$2
                     AND to_tsvector('english', line_text) @@ plainto_tsquery('english', $3)
                   ORDER BY key, line_number
                   LIMIT 30""",
                session_id, key, pattern,
            )
        else:
            hit_rows = await conn.fetch(
                """SELECT key, line_number
                   FROM agent_workspace
                   WHERE session_id=$1
                     AND to_tsvector('english', line_text) @@ plainto_tsquery('english', $2)
                   ORDER BY key, line_number
                   LIMIT 30""",
                session_id, pattern,
            )

        if not hit_rows:
            return json.dumps({"pattern": pattern, "matches": [], "total": 0})

        # Для каждого совпадения — дотянуть контекстные строки
        results = []
        for hit in hit_rows:
            ctx_start = max(1, hit["line_number"] - context_lines)
            ctx_end   = hit["line_number"] + context_lines
            ctx_rows  = await conn.fetch(
                """SELECT line_number, line_text
                   FROM agent_workspace
                   WHERE session_id=$1 AND key=$2
                     AND line_number BETWEEN $3 AND $4
                   ORDER BY line_number""",
                session_id, hit["key"], ctx_start, ctx_end,
            )
            context_text = "\n".join(
                f"{'>>>' if r['line_number'] == hit['line_number'] else '   '} "
                f"{r['line_number']:>4}: {r['line_text']}"
                for r in ctx_rows
            )
            results.append({
                "key": hit["key"],
                "line_number": hit["line_number"],
                "context": context_text,
            })

    return json.dumps({"pattern": pattern, "total": len(results), "matches": results},
                      ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

model = LiteLlm(model=_litellm_model)

root_agent = Agent(
    name="wix_rag_agent",
    model=model,
    description="Ассистент поддержки Wix с доступом к базе знаний Wix Help Center.",
    before_model_callback=_trim_history,
    instruction="""Ты — Wix Assistant, умный помощник по продуктам и настройке Wix.

Ты ведёшь живой диалог: переспрашиваешь когда неясно, помнишь контекст разговора,
отвечаешь кратко если вопрос простой и подробно если сложный.
Не начинай каждый ответ с поиска — сначала пойми что человек хочет.

## Когда искать, а когда нет

Отвечай БЕЗ поиска если:
- это уточнение или продолжение предыдущего ответа ("а как это выглядит?", "и что дальше?")
- ты уже нашёл нужную информацию в этой сессии
- вопрос о чём-то что ты уже объяснял

Ищи когда:
- новая тема или новый вопрос
- тебе нужны конкретные шаги/настройки которых ты не знаешь наверняка
- пользователь явно просит найти документацию

## Как вести диалог

- Если вопрос расплывчатый — уточни прежде чем искать: "Ты имеешь в виду X или Y?"
- Не пересказывай весь JSON из инструментов — выдели главное
- Если нашёл несколько статей — кратко объясни что в каждой, спроси что интересует больше
- Если что-то не нашлось — скажи честно и предложи альтернативный поиск
- Отвечай на языке пользователя

## Поисковые инструменты (используй по необходимости)

**search_by_titles** — быстрый обзор: какие статьи вообще есть по теме.
Хорош как первый шаг когда тема новая.

**search_by_chunks** — поиск внутри документов по содержанию.
Используй когда нужны конкретные шаги, параметры, инструкции.
Можно вызывать с уточнённым запросом прямо по теме ("как подключить домен", "настройка SEO").

**open_article(article_id)** — сохраняет полную статью в артефакт, показывает первые 80 строк.
Если в ответе есть поле "note" — статья длиннее, остаток читай через read_article_lines.

## Инструменты артефактов (для работы с открытыми статьями)

**list_saved_articles()** — список статей открытых в этой сессии.

**read_article_lines(artifact_name, start_line, end_line)** — читать статью с нужной строки.
Используй когда open_article вернул "note" об обрезке.

**search_in_article(artifact_name, pattern)** — regex/текстовый поиск по конкретной статье.
Удобно когда статья большая и нужно найти конкретный термин или раздел.

**search_in_articles(artifact_names, pattern)** — поиск по нескольким открытым статьям.

## Рабочее пространство (PostgreSQL workspace)

Персистентное хранилище для больших выводов инструментов — построчно, с поиском.
Данные живут в рамках сессии. Паттерн: сохранил → читай по частям → ищи.

**workspace_write(key, content)** — сохранить текст построчно. key — короткое имя (напр. "search_results_domain").

**workspace_list()** — список всех записей с количеством строк.

**workspace_read(key, start_line, end_line)** — читать строки с позиции. Возвращает has_more если есть ещё.

**workspace_search(pattern, key=None)** — fulltext поиск по workspace. Возвращает номера строк и контекст ±2 строки вокруг совпадения (как grep -n -C 2).

## Формат ответа

- Короткий вопрос → короткий ответ (2-5 предложений)
- Пошаговая инструкция → нумерованный список
- Всегда указывай источник в конце: *Источник: [название статьи]*
- Если не уверен — скажи об этом явно, не придумывай
""",
    tools=[
        search_by_titles,
        search_by_chunks,
        open_article,
        list_saved_articles,
        read_article_lines,
        search_in_article,
        search_in_articles,
        workspace_write,
        workspace_list,
        workspace_read,
        workspace_search,
        load_memory_tool,
        preload_memory_tool,
    ],
)
