"""ADK callbacks: spill tool results + compress history."""
from __future__ import annotations

from context_compressor import maybe_compress_llm_request
from workspace_storage import (
    PERSIST_THRESHOLD_CHARS,
    coerce_tool_response_text,
    maybe_persist,
)

# Не персистим чтение из workspace (избегаем рекурсии / двойного spill)
_SKIP_PERSIST_TOOLS = frozenset({
    "open_article",  # уже пишет в workspace через store_text
    "workspace_list",
    "workspace_read",
    "workspace_search",
    "load_memory",
    "preload_memory",
})


async def after_tool_persist(tool, args, tool_context, tool_response):
    """Hermes layer 2: большие ответы тулов → agent_workspace + stub."""
    if tool.name in _SKIP_PERSIST_TOOLS:
        return None

    text = coerce_tool_response_text(tool_response)
    if len(text) <= PERSIST_THRESHOLD_CHARS:
        return None

    session_id = tool_context._invocation_context.session.id
    stub = await maybe_persist(session_id, tool.name, text)
    if stub == text:
        return None

    if isinstance(tool_response, dict):
        out = dict(tool_response)
        out["result"] = stub
        return out
    return {"result": stub}


async def before_model_prepare(callback_context, llm_request) -> None:
    """Сжатие истории (Hermes-style) перед запросом к LLM."""
    await maybe_compress_llm_request(llm_request)
