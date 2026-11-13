"""Content Tool Calls — hook plugin for Octop / octop-harness.

Some local models (qwen2.5-coder, dmp-*, coder7b, ...) understand the tool
schema but answer with a plain-text JSON tool call instead of emitting native
OpenAI ``tool_calls``. Octop's agent loop (langchain ``create_agent``) only
routes to the tool node when ``AIMessage.tool_calls`` is non-empty, so those
models can never execute a tool.

This plugin registers a middleware whose ``after_model`` hook inspects the
last assistant message. When it carries no native tool calls but its text
contains a tool call in one of the common textual formats, the hook rewrites
the message so it exposes native ``tool_calls``. Models that already emit
native tool calls are untouched (the hook returns ``None`` immediately).

Supported textual formats:

* ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (Qwen style)
* ``[TOOL_CALLS] [...]`` (DeepSeek style)
* fenced ```` ```json ... ``` ```` blocks
* one or more bare JSON objects ``{"name": ..., "arguments": {...}}`` — a
  single object as the whole message, several concatenated objects, or a
  leading object followed by prose
* the DMP fine-tune shape ``{"thoughts": ..., "command": ..., "args": {...}}``

JSON payloads are located with :class:`json.JSONDecoder` ``raw_decode`` rather
than regular expressions, so nested braces, brackets and strings inside
arguments are handled correctly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, RemoveMessage

logger = logging.getLogger(__name__)

_DECODER = json.JSONDecoder()
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _content_to_text(content: Any) -> str:
    """Flatten langchain message content (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _decode_at(text: str, index: int) -> tuple[Any, int] | None:
    """``raw_decode`` a JSON value starting at *index*; ``None`` on failure."""
    try:
        value, end = _DECODER.raw_decode(text, index)
    except ValueError:
        return None
    return value, end


def _iter_items(payload: Any) -> Iterator[Any]:
    """Yield candidate call objects from a decoded payload (list or object)."""
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload


def _normalize_call(item: Any) -> dict[str, Any] | None:
    """Map a parsed JSON object to ``{"name": ..., "args": ...}`` when it looks like a tool call."""
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("command") or item.get("tool")
    args = item.get("arguments")
    if args is None:
        args = item.get("args")
    if args is None:
        args = item.get("parameters")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if not _NAME_RE.match(name):
        return None
    # Some models double-encode the arguments as a JSON string.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "args": args}


def _scan_tagged(text: str, opener: str, closer: str | None) -> list[Any]:
    """Decode JSON values that immediately follow each *opener* marker."""
    found: list[Any] = []
    cursor = 0
    while True:
        start = text.find(opener, cursor)
        if start == -1:
            return found
        decoded = _decode_at(text, _skip_ws(text, start + len(opener)))
        if decoded is None:
            cursor = start + len(opener)
            continue
        value, end = decoded
        found.append(value)
        if closer is None:
            cursor = end
            continue
        cursor = text.find(closer, end)
        if cursor == -1:
            return found
        cursor += len(closer)


def _scan_fenced(text: str) -> list[Any]:
    """Decode JSON values that appear at the start of a fenced code block."""
    found: list[Any] = []
    index = 0
    while True:
        fence = text.find("```", index)
        if fence == -1:
            return found
        rest = fence + 3
        # Skip an info string such as ``json`` or ``JSON``.
        while rest < len(text) and text[rest] in " \t":
            rest += 1
        info_start = rest
        while rest < len(text) and text[rest].isalpha():
            rest += 1
        if text[info_start:rest].lower() not in ("", "json"):
            index = fence + 3
            continue
        decoded = _decode_at(text, _skip_ws(text, rest))
        if decoded is None:
            index = fence + 3
            continue
        value, end = decoded
        found.append(value)
        index = end


def _scan_at_mention(text: str) -> tuple[list[Any], bool]:
    """Decode JSON objects that follow ``@mention`` tokens (Octop team dispatch).

    Handles ``@专家名 {json}`` and concatenations like ``@A {..} @B {..}``.
    Returns ``(values, is_only_mentions)`` where the flag is true only when
    the whole text is consumed by mention+JSON pairs (ignoring whitespace),
    so the original text can be dropped instead of shown next to the call.
    """
    found: list[Any] = []
    cursor = 0
    last_end = 0
    gap = False
    while True:
        start = text.find("@", cursor)
        if start == -1:
            break
        # @ must be at start or preceded by whitespace (skip emails / mid-word @).
        if start > 0 and not text[start - 1].isspace():
            cursor = start + 1
            continue
        i = start + 1
        # Skip the mention name (non-space, non-{ characters).
        while i < len(text) and not text[i].isspace() and text[i] != "{":
            i += 1
        i = _skip_ws(text, i)
        if i < len(text) and text[i] == "{":
            decoded = _decode_at(text, i)
            if decoded is not None:
                value, end = decoded
                if text[last_end:start].strip():
                    gap = True
                found.append(value)
                last_end = end
                cursor = end
                continue
        cursor = start + 1
    if text[last_end:].strip():
        gap = True
    return found, not gap


def _scan_bare(text: str) -> list[Any] | None:
    """Decode *text* as one or more concatenated JSON values.

    Returns ``None`` as soon as a non-whitespace character cannot be consumed
    by the decoder, i.e. the text is not *entirely* JSON.
    """
    values: list[Any] = []
    index = 0
    while True:
        index = _skip_ws(text, index)
        if index >= len(text):
            return values
        decoded = _decode_at(text, index)
        if decoded is None:
            return None
        value, index = decoded
        values.append(value)


def _parse_calls(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Extract tool calls from *text*.

    Returns ``(calls, content_is_only_calls)``. The second flag is true only
    when the whole message content was tool-call JSON, so the text can be
    dropped instead of shown to the user.
    """
    stripped = text.strip()
    if not stripped:
        return [], False

    calls: list[dict[str, Any]] = []

    def _collect(payloads: list[Any]) -> None:
        for payload in payloads:
            for item in _iter_items(payload):
                call = _normalize_call(item)
                if call is not None:
                    calls.append(call)

    # Octop team dispatch: @专家名 {json} — check first so the mention prefix
    # does not confuse the bare/leading JSON scanners below.
    mention_values, mention_only = _scan_at_mention(stripped)
    if mention_values:
        _collect(mention_values)
        if calls:
            return calls, mention_only

    _collect(_scan_tagged(stripped, "<tool_call>", "</tool_call>"))
    _collect(_scan_tagged(stripped, "[TOOL_CALLS]", None))
    _collect(_scan_fenced(stripped))

    if calls:
        return calls, False

    # Whole-message JSON: one object, or several concatenated objects.
    values = _scan_bare(stripped)
    if values:
        _collect(values)
        if calls:
            return calls, True

    # Leading JSON object followed by prose: still a tool call, but keep the
    # text so the prose stays visible.
    if stripped.startswith(("{", "[")):
        decoded = _decode_at(stripped, 0)
        if decoded is not None:
            _collect([decoded[0]])
            if calls:
                return calls, False

    return [], False


class ContentToolCallMiddleware(AgentMiddleware):
    """Re-emit text-embedded tool calls as native ``tool_calls``."""

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") or []
        if not messages:
            return None

        last: AIMessage | None = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last = msg
                break
        if last is None or last.tool_calls:
            return None

        text = _content_to_text(last.content)
        calls, only_calls = _parse_calls(text)
        if not calls:
            return None

        logger.debug(
            "[content-tool-calls] parsed %d call(s) (only_calls=%s) from %d chars",
            len(calls),
            only_calls,
            len(text),
        )
        tool_calls: list[Any] = [
            {
                "id": f"call_{index}",
                "name": call["name"],
                "args": call["args"],
                "type": "tool_call",
            }
            for index, call in enumerate(calls)
        ]

        new_message = AIMessage(
            content="" if only_calls else last.content,
            tool_calls=tool_calls,
            name=last.name,
            response_metadata=last.response_metadata,
        )
        if last.id is not None:
            return {"messages": [RemoveMessage(id=last.id), new_message]}
        # No stable id to remove: mutate in place so the edge function that
        # routes to the tool node observes the synthesized tool calls.
        last.tool_calls = tool_calls
        if only_calls:
            last.content = ""
        return None


def setup(ctx: Any) -> None:
    ctx.middleware(ContentToolCallMiddleware(), priority=900)
    logger.info("[content-tool-calls] middleware registered (priority=900)")
