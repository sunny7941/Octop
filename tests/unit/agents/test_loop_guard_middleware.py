"""LoopGuard agent middleware tests."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop.infra.agents.middleware.loop_guard import (
    DEFAULT_MAX_REPEATS,
    DEFAULT_MAX_SAME_RESULTS,
    LoopGuardMiddleware,
    _call_key,
    _results_equivalent,
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


def _job_payload(identifier: str) -> str:
    """A cron-job result whose only variation is the generated id.

    Mirrors the real loop in thread ``thr_01M79CRGRWKHCXGF2KZVEKN5YM``.
    """
    return json.dumps(
        {
            "mode": "background",
            "agent_id": "61Z9JX",
            "name": "工程专家",
            "job_id": identifier,
            "short_id": identifier[-6:],
            "status": "queued",
            "message": "Background task started. Tell the user it is running.",
        },
        ensure_ascii=False,
    )


def _payload_passthrough(request: ToolCallRequest) -> ToolMessage:
    args = request.tool_call.get("args") or {}
    return ToolMessage(
        content=str(args.get("content") or ""),
        tool_call_id=str(request.tool_call.get("id") or ""),
    )


async def _apass_payload(request: ToolCallRequest) -> ToolMessage:
    return _payload_passthrough(request)


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


def test_duplicate_results_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool answering the same way twice is refused on the next call.

    The arguments differ each time, so only the duplicate-result rule can
    stop this loop.
    """
    _patch_thread(monkeypatch, "thr-dup-1")
    middleware = LoopGuardMiddleware()

    for identifier in ("6eefc2dda6ff401a9fac224f7ff3d93f", "b54725f5d5e4432c8e63f6e38b038a07"):
        result = middleware.wrap_tool_call(
            _request("ask_agent", {"agent_id": "61Z9JX", "content": _job_payload(identifier)}),
            _payload_passthrough,
        )
        assert not isinstance(result, ToolMessage) or result.status != "error"

    blocked = middleware.wrap_tool_call(
        _request("ask_agent", {"agent_id": "61Z9JX", "content": _job_payload("c0ffee1234567890")}),
        _payload_passthrough,
    )
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert "ask_agent" in blocked.content
    assert "near-identical" in blocked.content


def test_materially_different_results_do_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different payloads reset the streak — a burst of real reads is not a loop."""
    _patch_thread(monkeypatch, "thr-dup-2")
    middleware = LoopGuardMiddleware()

    for index in range(6):
        body = f"def handler_{index}(request):\n    return Response({index})\n" * 3
        result = middleware.wrap_tool_call(
            _request("read_file", {"path": f"/repo/mod_{index}.py", "content": body}),
            _payload_passthrough,
        )
        assert not isinstance(result, ToolMessage) or result.status != "error"


def test_duplicate_results_require_consecutive_same_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleaving another tool breaks the run."""
    _patch_thread(monkeypatch, "thr-dup-3")
    middleware = LoopGuardMiddleware()

    for index in range(4):
        for name in ("cronjob_create", "agent_list"):
            result = middleware.wrap_tool_call(
                _request(name, {"content": _job_payload(f"id-{index}")}),
                _payload_passthrough,
            )
            assert not isinstance(result, ToolMessage) or result.status != "error"


def test_short_results_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terse acknowledgement carries too little signal to judge."""
    _patch_thread(monkeypatch, "thr-dup-4")
    middleware = LoopGuardMiddleware()

    for _ in range(DEFAULT_MAX_SAME_RESULTS + 2):
        result = middleware.wrap_tool_call(_request("execute", {"command": "true"}), _passthrough)
        assert not isinstance(result, ToolMessage) or result.status != "error"


@pytest.mark.asyncio
async def test_async_hook_blocks_duplicate_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_thread(monkeypatch, "thr-dup-5")
    middleware = LoopGuardMiddleware()

    for identifier in ("6eefc2dda6ff401a9fac224f7ff3d93f", "b54725f5d5e4432c8e63f6e38b038a07"):
        await middleware.awrap_tool_call(
            _request("ask_agent", {"agent_id": "61Z9JX", "content": _job_payload(identifier)}),
            _apass_payload,
        )

    blocked = await middleware.awrap_tool_call(
        _request("ask_agent", {"agent_id": "61Z9JX", "content": _job_payload("c0ffee1234567890")}),
        _apass_payload,
    )
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"


def test_before_agent_resets_duplicate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh turn must not inherit the previous turn's loop history."""
    _patch_thread(monkeypatch, "thr-dup-6")
    middleware = LoopGuardMiddleware()

    for identifier in ("6eefc2dda6ff401a9fac224f7ff3d93f", "b54725f5d5e4432c8e63f6e38b038a07"):
        middleware.wrap_tool_call(
            _request("ask_agent", {"content": _job_payload(identifier)}),
            _payload_passthrough,
        )

    middleware.before_agent({}, None)

    result = middleware.wrap_tool_call(
        _request("ask_agent", {"content": _job_payload("c0ffee1234567890")}),
        _payload_passthrough,
    )
    assert not isinstance(result, ToolMessage) or result.status != "error"


def test_results_equivalent_units() -> None:
    """Only generated ids may differ; any real change counts as progress."""
    assert not _results_equivalent("ok", "ok")
    assert not _results_equivalent("", _job_payload("x"))
    assert _results_equivalent(
        _job_payload("6eefc2dda6ff401a9fac224f7ff3d93f"),
        _job_payload("b54725f5d5e4432c8e63f6e38b038a07"),
    )

    # A removed field is a real change, not a generated id.
    left = _job_payload("6eefc2dda6ff401a9fac224f7ff3d93f")
    right = left.replace("Background task started. Tell the user it is running.", "")
    assert not _results_equivalent(left, right)

    # A status transition is progress.
    running = _job_payload("6eefc2dda6ff401a9fac224f7ff3d93f").replace(
        '"status": "queued"', '"status": "running"'
    )
    done = _job_payload("b54725f5d5e4432c8e63f6e38b038a07").replace(
        '"status": "queued"', '"status": "done"'
    )
    assert not _results_equivalent(running, done)

    # A payload that is nothing but opaque tokens cannot be judged.
    blob = "aB3xK9mN2pQ7rT4vW8yZ1cD5eF6gH0jL" * 4
    assert not _results_equivalent(blob, blob[:-7] + "ZZZZZZZ")
