"""LoopGuard agent middleware tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop.infra.agents.middleware.loop_guard import (
    DEFAULT_MAX_REPEATS,
    LoopGuardMiddleware,
    _call_key,
)


def _request(name: str, args: dict[str, Any], call_id: str = "call-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id},
        tool=None,
        state={},
        runtime=None,
    )


def _patch_thread(monkeypatch: pytest.MonkeyPatch, thread_id: str) -> None:
    monkeypatch.setattr(
        "octop.infra.agents.middleware.loop_guard.get_config",
        lambda: {"configurable": {"thread_id": thread_id}},
    )


def _passthrough(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ran", tool_call_id=str(request.tool_call.get("id") or ""))


async def _apass(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ran", tool_call_id=str(request.tool_call.get("id") or ""))


def test_identical_calls_pass_until_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_thread(monkeypatch, "thr-1")
    middleware = LoopGuardMiddleware(max_repeats=DEFAULT_MAX_REPEATS)

    for _ in range(DEFAULT_MAX_REPEATS - 1):
        result = middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
        assert not isinstance(result, ToolMessage) or result.status != "error"

    blocked = middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert "cronjob_create" in blocked.content
    assert str(DEFAULT_MAX_REPEATS) in blocked.content


def test_changed_arguments_restart_the_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_thread(monkeypatch, "thr-2")
    middleware = LoopGuardMiddleware(max_repeats=DEFAULT_MAX_REPEATS)

    for _ in range(DEFAULT_MAX_REPEATS - 1):
        middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    # A materially different call breaks the run.
    middleware.wrap_tool_call(_request("cronjob_create", {"name": "y"}), _passthrough)
    result = middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    assert not isinstance(result, ToolMessage) or result.status != "error"


def test_different_tool_name_restarts_the_streak(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_thread(monkeypatch, "thr-3")
    middleware = LoopGuardMiddleware(max_repeats=DEFAULT_MAX_REPEATS)

    for _ in range(DEFAULT_MAX_REPEATS - 1):
        middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    middleware.wrap_tool_call(_request("cronjob_delete", {"name": "x"}), _passthrough)
    result = middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    assert not isinstance(result, ToolMessage) or result.status != "error"


def test_streaks_are_tracked_per_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_thread(monkeypatch, "thr-4")
    middleware = LoopGuardMiddleware(max_repeats=DEFAULT_MAX_REPEATS)

    for _ in range(DEFAULT_MAX_REPEATS - 1):
        middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)

    # Same call on a different thread has its own counter.
    _patch_thread(monkeypatch, "thr-5")
    result = middleware.wrap_tool_call(_request("cronjob_create", {"name": "x"}), _passthrough)
    assert not isinstance(result, ToolMessage) or result.status != "error"


@pytest.mark.asyncio
async def test_async_hook_blocks_the_repeated_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_thread(monkeypatch, "thr-6")
    middleware = LoopGuardMiddleware(max_repeats=DEFAULT_MAX_REPEATS)

    for _ in range(DEFAULT_MAX_REPEATS - 1):
        await middleware.awrap_tool_call(_request("cronjob_create", {"name": "x"}), _apass)

    blocked = await middleware.awrap_tool_call(_request("cronjob_create", {"name": "x"}), _apass)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"


def test_call_key_ignores_argument_order() -> None:
    assert _call_key("t", {"a": 1, "b": 2}) == _call_key("t", {"b": 2, "a": 1})
    assert _call_key("t", {"a": 1}) != _call_key("t", {"a": 2})
    assert _call_key("t", {"a": 1}) != _call_key("u", {"a": 1})
