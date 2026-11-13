"""Tests for the bundled content-tool-calls plugin.

The plugin re-emits plain-text JSON tool calls from local models as native
OpenAI ``tool_calls`` so the agent loop can route to the tool node. These tests
cover the textual formats it recognises and the ``after_model`` rewrite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

_PLUGIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "octop"
    / "infra"
    / "agents"
    / "plugins"
    / "bundled"
    / "content-tool-calls"
)
_MODULE_NAME = "octop_test_content_tool_calls"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PLUGIN_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


def _names(calls: list[dict[str, Any]]) -> list[str]:
    return [call["name"] for call in calls]


# ── _parse_calls: recognised formats ─────────────────────────────────


def test_parse_bare_single_object(mod: Any) -> None:
    calls, only = mod._parse_calls('{"name": "get_weather", "arguments": {"city": "北京"}}')
    assert _names(calls) == ["get_weather"]
    assert calls[0]["args"] == {"city": "北京"}
    assert only is True


def test_parse_tagged_tool_call_keeps_surrounding_text(mod: Any) -> None:
    text = '好的，我来查。\n<tool_call>{"name": "get_weather", "arguments": {"city": "上海"}}</tool_call>'
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["get_weather"]
    assert only is False


def test_parse_fenced_json_block(mod: Any) -> None:
    text = '```json\n{"name": "run_command", "arguments": {"command": "ls"}}\n```'
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["run_command"]
    assert only is False


def test_parse_leading_json_followed_by_prose(mod: Any) -> None:
    text = '{"name": "read_file", "arguments": {"path": "a.py"}} 已读取文件'
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["read_file"]
    assert only is False


def test_parse_plain_prose_yields_no_call(mod: Any) -> None:
    calls, only = mod._parse_calls("今天天气不错，适合出门。")
    assert calls == []
    assert only is False


def test_parse_nested_arguments_with_braces_and_strings(mod: Any) -> None:
    payload = (
        '{"name": "write_file", "arguments": {"path": "a.py", '
        '"content": "print({\\"a\\": [1, 2]})"}}'
    )
    calls, _ = mod._parse_calls(payload)
    assert _names(calls) == ["write_file"]
    assert calls[0]["args"]["content"] == 'print({"a": [1, 2]})'


def test_parse_double_encoded_arguments_string(mod: Any) -> None:
    text = '{"name": "get_weather", "arguments": "{\\"city\\": \\"北京\\"}"}'
    calls, _ = mod._parse_calls(text)
    assert calls[0]["args"] == {"city": "北京"}


def test_parse_dmp_command_shape(mod: Any) -> None:
    text = '{"thoughts": "需要查天气", "command": "get_weather", "args": {"city": "北京"}}'
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["get_weather"]
    assert calls[0]["args"] == {"city": "北京"}
    assert only is True


def test_parse_at_mention_team_dispatch(mod: Any) -> None:
    text = (
        '@数据分析师 {"name": "ask_agent", '
        '"arguments": {"expert": "数据分析师", "message": "汇总销售数据"}}'
    )
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["ask_agent"]
    assert calls[0]["args"]["expert"] == "数据分析师"
    assert only is True


def test_parse_at_mention_keeps_prose_before(mod: Any) -> None:
    text = (
        '我帮你转给专家。\n@数据分析师 {"name": "ask_agent", '
        '"arguments": {"expert": "数据分析师", "message": "汇总销售数据"}}'
    )
    calls, only = mod._parse_calls(text)
    assert _names(calls) == ["ask_agent"]
    assert only is False


def test_parse_rejects_non_tool_json(mod: Any) -> None:
    """A JSON answer without a name/arguments pair is not a tool call."""
    calls, only = mod._parse_calls('{"city": "北京", "temp": 21}')
    assert calls == []
    assert only is False


# ── after_model rewrite ──────────────────────────────────────────────


def _state(text: str, *, message_id: str | None = None) -> dict[str, Any]:
    return {"messages": [AIMessage(content=text, id=message_id)]}


def test_after_model_rewrites_text_call_as_native(mod: Any) -> None:
    mw = mod.ContentToolCallMiddleware()
    state = _state('{"name": "get_weather", "arguments": {"city": "北京"}}')
    update = mw.after_model(state, None)
    assert update is None  # no stable id -> in-place mutation
    msg = state["messages"][-1]
    assert [call["name"] for call in msg.tool_calls] == ["get_weather"]
    assert msg.tool_calls[0]["args"] == {"city": "北京"}
    assert msg.content == ""  # whole message was the call


def test_after_model_with_id_returns_remove_and_new_message(mod: Any) -> None:
    mw = mod.ContentToolCallMiddleware()
    state = _state(
        '{"name": "get_weather", "arguments": {"city": "北京"}}',
        message_id="msg-1",
    )
    update = mw.after_model(state, None)
    assert update is not None
    kinds = [type(item).__name__ for item in update["messages"]]
    assert kinds == ["RemoveMessage", "AIMessage"]


def test_after_model_leaves_native_tool_calls_untouched(mod: Any) -> None:
    mw = mod.ContentToolCallMiddleware()
    native = AIMessage(
        content="",
        tool_calls=[{"id": "call_0", "name": "get_weather", "args": {}, "type": "tool_call"}],
    )
    state = {"messages": [native]}
    assert mw.after_model(state, None) is None
    assert state["messages"][-1] is native


def test_after_model_ignores_plain_answer(mod: Any) -> None:
    mw = mod.ContentToolCallMiddleware()
    state = _state("你好，有什么可以帮你的？")
    assert mw.after_model(state, None) is None
    assert state["messages"][-1].tool_calls == []


def test_after_model_uses_last_assistant_message(mod: Any) -> None:
    mw = mod.ContentToolCallMiddleware()
    state = {
        "messages": [
            AIMessage(content="早前的问题"),
            AIMessage(content='{"name": "ping", "arguments": {}}'),
        ],
    }
    mw.after_model(state, None)
    assert state["messages"][-1].tool_calls[0]["name"] == "ping"
