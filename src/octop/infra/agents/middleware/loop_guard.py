"""Break degenerate tool-call loops before a local model can spin for hours.

A weak local model sometimes re-issues the same tool call with byte-identical
arguments forever — Octop once created 332 identical cron jobs in one turn
because nothing stopped it. ``recursion_limit`` only caps graph steps, so a
loop of cheap calls can run for hours before anyone notices.

This middleware counts consecutive calls that share a ``(name, args)`` key
with no different call in between, and answers the repeated call with an error
message that tells the model to change strategy or stop and report. It never
raises: the turn stays alive and the model gets a chance to recover.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

DEFAULT_MAX_REPEATS = 5
# Args are hashed, so a pathological payload cannot grow the key without bound.
_MAX_ARGS_CHARS = 2000
# Bound the per-agent table of abandoned threads.
_MAX_TRACKED_THREADS = 256

# Fallback bucket when no thread id is visible (e.g. a one-shot ACP call).
_ANONYMOUS_THREAD = "__anonymous__"


def _current_thread_id() -> str:
    try:
        configurable = dict(get_config().get("configurable") or {})
    except RuntimeError:
        return ""
    raw = configurable.get("thread_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _ANONYMOUS_THREAD


def _call_key(name: str, args: Any) -> str:
    """Stable identity for a tool call, insensitive to JSON key order."""
    try:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = repr(args)
    if len(payload) > _MAX_ARGS_CHARS:
        payload = payload[:_MAX_ARGS_CHARS]
    return hashlib.sha256(f"{name}\x00{payload}".encode()).hexdigest()


def _notice(tool_name: str, repeats: int) -> str:
    return (
        f"LoopGuard: `{tool_name}` has now been called {repeats} times in a row with "
        "identical arguments, so this call was refused. Repeating it will not make "
        "progress. Do NOT call it again with the same arguments — change the arguments "
        "materially, switch to a different tool, or stop and report the current status "
        "to the user."
    )


class LoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Refuse the Nth consecutive identical tool call within a turn."""

    def __init__(self, *, max_repeats: int = DEFAULT_MAX_REPEATS) -> None:
        super().__init__()
        self._max_repeats = max_repeats if max_repeats > 0 else DEFAULT_MAX_REPEATS
        # thread_id -> [last key, consecutive repeats]
        self._streaks: dict[str, list[Any]] = {}

    def before_agent(self, state: Any, runtime: Any) -> None:
        del state, runtime
        self._reset()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        del state, runtime
        self._reset()

    def _reset(self) -> None:
        thread_id = _current_thread_id()
        self._streaks.pop(thread_id, None)

    def _streak(self, thread_id: str, key: str) -> int:
        entry = self._streaks.get(thread_id)
        if entry is None:
            if len(self._streaks) >= _MAX_TRACKED_THREADS:
                self._streaks.clear()
            self._streaks[thread_id] = [key, 1]
            return 1
        if entry[0] == key:
            entry[1] += 1
            return int(entry[1])
        entry[0] = key
        entry[1] = 1
        return 1

    def _guard(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_call = request.tool_call
        name = str(tool_call.get("name") or "")
        if not name:
            return None
        args = tool_call.get("args")
        if not isinstance(args, (dict, list)):
            args = {}
        repeats = self._streak(_current_thread_id(), _call_key(name, args))
        if repeats < self._max_repeats:
            return None
        logger.info(
            "LoopGuard refused repeat %d of %s (thread=%s)",
            repeats,
            name,
            _current_thread_id(),
        )
        return ToolMessage(
            content=_notice(name, repeats),
            tool_call_id=str(tool_call.get("id") or ""),
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if (notice := self._guard(request)) is not None:
            return notice
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if (notice := self._guard(request)) is not None:
            return notice
        return await handler(request)


__all__ = [
    "DEFAULT_MAX_REPEATS",
    "LoopGuardMiddleware",
    "_call_key",
    "_notice",
]
