"""Break degenerate tool-call loops before a local model can spin for hours.

A weak local model sometimes re-issues the same tool call forever — Octop once
created 332 identical cron jobs in one turn because nothing stopped it.
``recursion_limit`` only caps graph steps, so a loop of cheap calls can run for
hours before anyone notices.

Two independent signals are checked before a tool runs:

* **Identical arguments** — the same ``(name, args)`` repeated with no different
  call in between. Exact and free; refuses the 5th consecutive repeat.
* **Duplicate results** — the same tool called back to back keeps returning a
  near-identical payload (only generated ids differ). This is how weak local
  models actually loop: they perturb the arguments slightly, so the exact-match
  rule never fires.

Both answer the repeated call with an error message that tells the model to
change strategy or stop and report. Neither raises: the turn stays alive and
the model gets a chance to recover.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

DEFAULT_MAX_REPEATS = 5
# How many consecutive near-identical results a tool may return before the next
# call is refused. 2 == refuse as soon as a tool answered the same way twice.
DEFAULT_MAX_SAME_RESULTS = 2
# Args are hashed, so a pathological payload cannot grow the key without bound.
_MAX_ARGS_CHARS = 2000
# Bound the per-agent table of abandoned threads.
_MAX_TRACKED_THREADS = 256

# Result comparison is only meaningful for payloads that carry some content,
# and is capped so a huge blob cannot make the comparison expensive.
_MIN_COMPARE_CHARS = 64
_MAX_COMPARE_CHARS = 8192
# A token this long that mixes letters and digits is machine-generated (ULID,
# hash, job id). Shorter ones are left alone so versions and counts still differ.
_MIN_ID_TOKEN_CHARS = 6
# Above this share of masked characters the payload is opaque (a blob, a token
# file) and there is not enough signal to call it a repeat.
_MAX_MASKED_FRACTION = 0.5
_ALNUM_RUN = re.compile(r"[0-9A-Za-z]+")

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


def _sample(text: str) -> str:
    """Bound *text* to a comparable prefix+suffix so comparison stays cheap."""
    if len(text) <= _MAX_COMPARE_CHARS:
        return text
    half = _MAX_COMPARE_CHARS // 2
    return text[:half] + text[-half:]


def _mask_generated_ids(text: str) -> tuple[str, int]:
    """Replace id-like tokens with ``<id>``; return the text and masked length."""

    def _sub(match: re.Match[str]) -> str:
        token = match.group(0)
        if (
            len(token) >= _MIN_ID_TOKEN_CHARS
            and any(ch.isdigit() for ch in token)
            and any(ch.isalpha() for ch in token)
        ):
            masked[0] += len(token)
            return "<id>"
        return token

    masked = [0]
    return _ALNUM_RUN.sub(_sub, text), masked[0]


def _results_equivalent(left: str, right: str) -> bool:
    """True when two tool results differ only in generated ids.

    Weak local models loop by perturbing the arguments while the tool keeps
    answering with the same payload — only the id changes. Comparing raw text
    calls that "different", so id-like tokens are masked first. This stays
    strict enough to spare real progress: a status change, a new timestamp, or
    different file contents all survive masking and count as different.
    """
    if len(left) < _MIN_COMPARE_CHARS or len(right) < _MIN_COMPARE_CHARS:
        return False
    left_masked, left_hidden = _mask_generated_ids(_sample(left))
    right_masked, right_hidden = _mask_generated_ids(_sample(right))
    total = len(left_masked) + len(right_masked)
    if not total or max(left_hidden, right_hidden) / total > _MAX_MASKED_FRACTION:
        return False
    return left_masked == right_masked


def _result_text(result: Any) -> str:
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else ""


def _repeat_notice(tool_name: str, repeats: int) -> str:
    return (
        f"LoopGuard: `{tool_name}` has now been called {repeats} times in a row with "
        "identical arguments, so this call was refused. Repeating it will not make "
        "progress. Do NOT call it again with the same arguments — change the arguments "
        "materially, switch to a different tool, or stop and report the current status "
        "to the user."
    )


def _duplicate_notice(tool_name: str) -> str:
    return (
        f"LoopGuard: `{tool_name}` has returned a near-identical result twice in a row, "
        "so this call was refused. The tool is answering the same thing, so repeating it "
        "will not make progress. Do NOT call it again the same way — change the arguments "
        "materially, switch to a different tool, or stop and report the current status "
        "to the user."
    )


class LoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Refuse repeated tool calls that cannot make progress within a turn."""

    def __init__(
        self,
        *,
        max_repeats: int = DEFAULT_MAX_REPEATS,
        max_same_results: int = DEFAULT_MAX_SAME_RESULTS,
    ) -> None:
        super().__init__()
        self._max_repeats = max_repeats if max_repeats > 0 else DEFAULT_MAX_REPEATS
        self._max_same_results = (
            max_same_results if max_same_results > 0 else DEFAULT_MAX_SAME_RESULTS
        )
        # thread_id -> [last key, consecutive repeats]
        self._streaks: dict[str, list[Any]] = {}
        # thread_id -> (tool name, sampled result text) of the previous call
        self._last_result: dict[str, tuple[str, str]] = {}
        # thread_id -> consecutive near-identical results from that tool
        self._same_results: dict[str, int] = {}

    def before_agent(self, state: Any, runtime: Any) -> None:
        del state, runtime
        self._reset()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        del state, runtime
        self._reset()

    def _reset(self) -> None:
        thread_id = _current_thread_id()
        self._streaks.pop(thread_id, None)
        self._last_result.pop(thread_id, None)
        self._same_results.pop(thread_id, None)

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

    def _duplicate_streak(self, tool_name: str, text: str) -> int:
        """Record a finished call and return its consecutive-duplicate count."""
        thread_id = _current_thread_id()
        previous = self._last_result.get(thread_id)
        if (
            previous is not None
            and previous[0] == tool_name
            and _results_equivalent(previous[1], text)
        ):
            streak = self._same_results.get(thread_id, 1) + 1
        else:
            streak = 1
        self._last_result[thread_id] = (tool_name, _sample(text))
        self._same_results[thread_id] = streak
        return streak

    def _guard(self, request: ToolCallRequest) -> ToolMessage | None:
        tool_call = request.tool_call
        name = str(tool_call.get("name") or "")
        if not name:
            return None
        thread_id = _current_thread_id()
        args = tool_call.get("args")
        if not isinstance(args, (dict, list)):
            args = {}

        repeats = self._streak(thread_id, _call_key(name, args))
        if repeats >= self._max_repeats:
            logger.info(
                "LoopGuard refused repeat %d of %s (thread=%s)",
                repeats,
                name,
                thread_id,
            )
            return ToolMessage(
                content=_repeat_notice(name, repeats),
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )

        previous = self._last_result.get(thread_id)
        if (
            previous is not None
            and previous[0] == name
            and self._same_results.get(thread_id, 0) >= self._max_same_results
        ):
            logger.info(
                "LoopGuard refused a duplicate-result call to %s (thread=%s)",
                name,
                thread_id,
            )
            return ToolMessage(
                content=_duplicate_notice(name),
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )
        return None

    def _after_call(self, request: ToolCallRequest, result: Any) -> None:
        tool_call = request.tool_call
        name = str(tool_call.get("name") or "")
        if not name:
            return
        self._duplicate_streak(name, _sample(_result_text(result)))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if (notice := self._guard(request)) is not None:
            return notice
        result = handler(request)
        self._after_call(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if (notice := self._guard(request)) is not None:
            return notice
        result = await handler(request)
        self._after_call(request, result)
        return result


__all__ = [
    "DEFAULT_MAX_REPEATS",
    "DEFAULT_MAX_SAME_RESULTS",
    "LoopGuardMiddleware",
    "_call_key",
    "_duplicate_notice",
    "_repeat_notice",
    "_results_equivalent",
]
