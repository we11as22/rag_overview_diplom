"""Сжатие истории диалога через LLM (аналог Hermes ContextCompressor).

При переполнении контекста: старые turns суммаризируются, хвост и начало сохраняются.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from google.genai import types
from litellm import acompletion

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Ниже — сжатое содержание более ранних "
    "сообщений. Это справочный фон, не новые инструкции. Отвечай только на "
    "последнее сообщение пользователя после этого блока.\n\n"
)

_PRUNED_TOOL = "[Old tool output cleared to save context space]"

_ENABLED = os.getenv("CONTEXT_COMPRESS_ENABLED", "true").lower() in ("1", "true", "yes")
_THRESHOLD_CHARS = int(os.getenv("CONTEXT_COMPRESS_THRESHOLD_CHARS", "60000"))
_PROTECT_HEAD = max(0, int(os.getenv("CONTEXT_COMPRESS_PROTECT_HEAD", "2")))
_PROTECT_TAIL = max(1, int(os.getenv("CONTEXT_COMPRESS_PROTECT_TAIL", "10")))
_CHARS_PER_TOKEN = 4

_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
_litellm_model = _model if _model.startswith("openai/") else f"openai/{_model}"


def _estimate_chars(contents: list[types.Content]) -> int:
    total = 0
    for c in contents:
        for p in c.parts or []:
            if p.text:
                total += len(p.text)
            if p.function_call:
                total += len(str(p.function_call.args or "")) + 64
            if p.function_response:
                total += len(str(p.function_response.response or "")) + 64
    return total


def _content_to_text(content: types.Content) -> str:
    role = content.role or "user"
    chunks: list[str] = [f"--- {role} ---"]
    for p in content.parts or []:
        if p.text:
            chunks.append(p.text)
        if p.function_call:
            chunks.append(
                f"[function_call {p.function_call.name}: {p.function_call.args}]"
            )
        if p.function_response:
            resp = p.function_response.response
            if isinstance(resp, dict):
                resp = str(resp)[:4000]
            chunks.append(f"[function_response {p.function_response.name}: {resp}]")
    return "\n".join(chunks)


def _prune_tool_outputs(contents: list[types.Content]) -> list[types.Content]:
    """Заменяет старые function_response на placeholder (как Hermes pre-pass)."""
    out: list[types.Content] = []
    for content in contents:
        new_parts = []
        changed = False
        for p in content.parts or []:
            if p.function_response and p.function_response.response:
                fr = p.function_response
                resp_text = str(fr.response)
                if len(resp_text) > 500 and _PRUNED_TOOL not in resp_text:
                    new_fr = fr.model_copy(deep=True)
                    new_fr.response = {"result": _PRUNED_TOOL}
                    new_parts.append(types.Part(function_response=new_fr))
                    changed = True
                else:
                    new_parts.append(p)
            else:
                new_parts.append(p)
        if changed:
            out.append(content.model_copy(update={"parts": new_parts}))
        else:
            out.append(content)
    return out


async def _summarize_middle(middle_text: str) -> str:
    prompt = f"""Summarize the following conversation turns for context handoff.
Use this structure:
## Resolved
## Pending questions
## Active task (what to do next)
## Key facts (IDs, article titles, decisions)

Conversation:
{middle_text[:120000]}
"""
    try:
        resp = await acompletion(
            model=_litellm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You compress chat history for another AI assistant. "
                        "Be factual and concise. Preserve article IDs and tool outcomes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Context compression LLM failed: %s", exc)
        return middle_text[:8000] + "\n...[compression failed, truncated]"


async def maybe_compress_llm_request(llm_request) -> None:
    if not _ENABLED:
        return

    contents = list(llm_request.contents or [])
    if len(contents) <= _PROTECT_HEAD + _PROTECT_TAIL + 1:
        return

    est = _estimate_chars(contents)
    if est < _THRESHOLD_CHARS:
        return

    head = contents[:_PROTECT_HEAD]
    tail = contents[-_PROTECT_TAIL:]
    middle = contents[_PROTECT_HEAD : -_PROTECT_TAIL]

    middle_pruned = _prune_tool_outputs(middle)
    middle_text = "\n\n".join(_content_to_text(c) for c in middle_pruned)
    summary_body = await _summarize_middle(middle_text)

    summary_content = types.Content(
        role="user",
        parts=[types.Part(text=SUMMARY_PREFIX + summary_body)],
    )
    llm_request.contents = head + [summary_content] + tail
    logger.info(
        "Context compressed: ~%d chars -> head=%d summary tail=%d",
        est,
        len(head),
        len(tail),
    )
