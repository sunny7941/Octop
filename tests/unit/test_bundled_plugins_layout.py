from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

import octop.infra.market.http as market_http
from octop.infra.agents.plugins.bundled import default_bundled_plugins_root
from octop.infra.agents.plugins.manager import PluginManager
from octop.infra.agents.plugins.seed import seed_bundled_plugins
from octop.infra.errors import ErrorCode, OctopError
from octop.infra.market import PortalMarketError, clear_market_cache


class _FakeSettings:
    def __init__(self, data: dict[str, str] | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value


def _catalog_ids() -> set[str]:
    root = default_bundled_plugins_root()
    found: set[str] = set()
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "plugin.yaml").is_file():
            found.add(child.name)
    return found


def test_in_tree_plugin_catalog_ships_bundled_plugins() -> None:
    assert _catalog_ids() == {"content-tool-calls", "egress-guard"}


def test_seed_installs_bundled_airgap_disabled(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    config_path = tmp_path / "config.json"
    copied = seed_bundled_plugins(
        bundled_root=default_bundled_plugins_root(),
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == ["content-tool-calls", "egress-guard"]
    assert (plugins_dir / "egress-guard" / "plugin.yaml").is_file()
    assert (plugins_dir / "content-tool-calls" / "plugin.yaml").is_file()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["plugins"]["egress-guard"]["enabled"] is False
    # content-tool-calls is a compatibility shim, on by default.
    assert raw["plugins"]["content-tool-calls"]["enabled"] is True


def test_seed_records_bundled_opt_out_in_settings(tmp_path: Path) -> None:
    """An existing DB has no flag yet; load_installed would default it to enabled."""
    mgr = PluginManager(plugins_dir=tmp_path / "plugins", config_path=tmp_path / "config.json")
    assert mgr.seed_bundled() == ["content-tool-calls", "egress-guard"]
    settings = _FakeSettings()
    mgr._plugin_settings = settings
    mgr.seed_bundled()
    flags = json.loads(settings.data["plugins.enabled"])
    assert flags["egress-guard"] is False
    assert flags["content-tool-calls"] is True


def test_upgrade_preserves_legacy_enabled_in_settings(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    dest = plugins_dir / "egress-guard"
    shutil.copytree(default_bundled_plugins_root() / "egress-guard", dest)
    manifest = (dest / "plugin.yaml").read_text(encoding="utf-8")
    (dest / "plugin.yaml").write_text(
        re.sub(r"(?m)^version: .*$", "version: 0.0.1", manifest),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"plugins": {"egress-guard": {"enabled": True}}}),
        encoding="utf-8",
    )
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=config_path)
    settings = _FakeSettings()
    mgr._plugin_settings = settings
    assert mgr.seed_bundled() == ["content-tool-calls", "egress-guard"]
    flag = json.loads(settings.data["plugins.enabled"])["egress-guard"]
    assert flag is True


def test_install_path_copies_fixture_plugin(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    config_path = tmp_path / "config.json"
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=config_path)
    sample = Path(__file__).resolve().parents[1] / "fixtures" / "plugins" / "echo-tool"
    loaded = mgr.install_path(sample)
    assert loaded.manifest.id == sample.name
    assert (plugins_dir / sample.name / "plugin.yaml").is_file()


def test_list_market_does_not_fall_back_to_bundled_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTOP_PORTAL_ORIGIN", "")
    clear_market_cache()
    mgr = PluginManager(plugins_dir=tmp_path / "plugins", config_path=tmp_path / "config.json")
    with pytest.raises(OctopError) as raised:
        mgr.list_market()
    assert raised.value.code is ErrorCode.PLUGIN_MARKET_FAILED


def test_list_market_network_failure_is_not_a_market_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTOP_PORTAL_ORIGIN", "https://octop.cloud")
    clear_market_cache()

    def fake_fetch(url: str, *, limit: int) -> bytes:
        del url, limit
        raise PortalMarketError("unreachable: down", network=True)

    monkeypatch.setattr(market_http, "fetch_bytes", fake_fetch)
    mgr = PluginManager(plugins_dir=tmp_path / "plugins", config_path=tmp_path / "config.json")
    with pytest.raises(OctopError) as raised:
        mgr.list_market()
    assert raised.value.code is ErrorCode.NETWORK_ERROR


def test_install_from_market_does_not_copy_bundled_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCTOP_PORTAL_ORIGIN", "https://octop.cloud")
    clear_market_cache()

    def fake_fetch(url: str, *, limit: int) -> bytes:
        del url, limit
        return json.dumps({"success": True, "data": {"items": []}}).encode()

    monkeypatch.setattr(market_http, "fetch_bytes", fake_fetch)
    plugins_dir = tmp_path / "plugins"
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=tmp_path / "config.json")
    with pytest.raises(OctopError) as raised:
        mgr.install_from_market("fortune")
    assert raised.value.code is ErrorCode.NOT_FOUND
    assert not (plugins_dir / "fortune").exists()
