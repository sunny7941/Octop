"""Unit tests for bundled plugin seeding (upgrade-only; market installs new)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from octop_harness.plugins import PluginRegistry

from octop.infra.agents.plugins.manager import PluginManager
from octop.infra.agents.plugins.seed import seed_bundled_plugins
from octop.infra.errors import ErrorCode, OctopError


def _write_plugin(root: Path, plugin_id: str, *, version: str = "0.1.0") -> Path:
    dest = root / plugin_id
    dest.mkdir(parents=True)
    (dest / "plugin.yaml").write_text(
        f"id: {plugin_id}\nversion: {version}\nname: {plugin_id}\nkind: tool\nentry: main.py\n",
        encoding="utf-8",
    )
    (dest / "main.py").write_text(
        "from octop_harness.plugins import PluginContext\n"
        "\n"
        "def setup(ctx: PluginContext) -> None:\n"
        f"    ctx.tool('{plugin_id}_tool', lambda: 'ok', description='d')\n",
        encoding="utf-8",
    )
    return dest


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


def test_seed_does_not_auto_install_missing(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather")
    plugins_dir = tmp_path / "plugins"
    config_path = tmp_path / "config.json"
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == []
    assert not (plugins_dir / "weather").exists()


def test_seed_does_not_overwrite_same_version(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather", version="1.0.0")
    (bundled / "weather" / "main.py").write_text("NEW = 1\n", encoding="utf-8")
    plugins_dir = tmp_path / "plugins"
    dest = _write_plugin(plugins_dir, "weather", version="1.0.0")
    (dest / "main.py").write_text("OLD = 1\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == []
    assert "OLD" in (dest / "main.py").read_text(encoding="utf-8")


def test_seed_refreshes_when_bundled_version_newer(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather", version="2.0.0")
    (bundled / "weather" / "marker.txt").write_text("new", encoding="utf-8")
    plugins_dir = tmp_path / "plugins"
    dest = _write_plugin(plugins_dir, "weather", version="1.0.0")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "plugins": {"weather": {"enabled": True}},
                "bundled_plugins_seeded": ["weather"],
            },
        ),
        encoding="utf-8",
    )
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == ["weather"]
    assert "2.0.0" in (dest / "plugin.yaml").read_text(encoding="utf-8")
    assert (dest / "marker.txt").read_text(encoding="utf-8") == "new"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["plugins"]["weather"]["enabled"] is True


def test_seed_does_not_recreate_after_uninstall(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather")
    plugins_dir = tmp_path / "plugins"
    _write_plugin(plugins_dir, "weather")
    config_path = tmp_path / "config.json"
    seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    shutil.rmtree(plugins_dir / "weather")
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == []
    assert not (plugins_dir / "weather").exists()


def test_seed_does_not_add_new_bundled_id(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather")
    plugins_dir = tmp_path / "plugins"
    _write_plugin(plugins_dir, "weather")
    config_path = tmp_path / "config.json"
    seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    _write_plugin(bundled, "qrcode")
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == []
    assert not (plugins_dir / "qrcode").exists()


def test_seed_installs_auto_install_plugin_disabled(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    dest = _write_plugin(bundled, "weather")
    (dest / "plugin.yaml").write_text(
        "id: weather\nversion: 0.1.0\nname: weather\nkind: hook\nentry: main.py\n"
        "auto_install: true\ndefault_enabled: false\n",
        encoding="utf-8",
    )
    plugins_dir = tmp_path / "plugins"
    config_path = tmp_path / "config.json"
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    assert copied == ["weather"]
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["plugins"]["weather"]["enabled"] is False


def test_seed_does_not_reinstall_auto_plugin_after_uninstall(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    dest = _write_plugin(bundled, "weather")
    (dest / "plugin.yaml").write_text(
        "id: weather\nversion: 0.1.0\nname: weather\nkind: hook\nentry: main.py\n"
        "auto_install: true\ndefault_enabled: false\n",
        encoding="utf-8",
    )
    plugins_dir = tmp_path / "plugins"
    config_path = tmp_path / "config.json"
    seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
    )
    shutil.rmtree(plugins_dir / "weather")
    copied = seed_bundled_plugins(
        bundled_root=bundled,
        plugins_dir=plugins_dir,
        config_path=config_path,
        known_plugins={"weather"},
    )
    assert copied == []
    assert not (plugins_dir / "weather").exists()


def test_load_installed_skips_disabled_before_import(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    _write_plugin(plugins_dir, "weather")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"plugins": {"weather": {"enabled": False}}}),
        encoding="utf-8",
    )
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=config_path)
    loaded = mgr.load_installed(install_deps=False)
    assert loaded == []
    assert PluginRegistry().get("weather") is None


def test_load_installed_clears_stale_registry_entries(tmp_path: Path) -> None:
    other_dir = tmp_path / "other"
    _write_plugin(other_dir, "stale")
    PluginManager(plugins_dir=other_dir, config_path=tmp_path / "other.json").load_installed(
        install_deps=False,
    )
    assert PluginRegistry().get("stale") is not None

    plugins_dir = tmp_path / "plugins"
    _write_plugin(plugins_dir, "weather")
    loaded = PluginManager(
        plugins_dir=plugins_dir,
        config_path=tmp_path / "config.json",
    ).load_installed(install_deps=False)
    assert [item.manifest.id for item in loaded] == ["weather"]
    assert PluginRegistry().get("stale") is None
    assert PluginRegistry().get("weather") is not None


def test_seed_refuses_corrupt_config_and_preserves_bytes(tmp_path: Path) -> None:
    """issue #730: seeding always writes back, so a corrupt file must not read as ``{}``."""
    bundled = tmp_path / "bundled"
    _write_plugin(bundled, "weather")
    plugins_dir = tmp_path / "plugins"
    _write_plugin(plugins_dir, "weather")
    config_path = tmp_path / "config.json"
    original = '{\n  "bind_host": "0.0.0.0",\n  "database": {"driver": "postgresql"}\n,}'
    config_path.write_text(original, encoding="utf-8")
    with pytest.raises(OctopError) as excinfo:
        seed_bundled_plugins(
            bundled_root=bundled,
            plugins_dir=plugins_dir,
            config_path=config_path,
        )
    assert excinfo.value.code is ErrorCode.CONFIG_FILE_CORRUPT
    assert config_path.read_text(encoding="utf-8") == original


def test_load_installed_defers_hook_plugins(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    dest = _write_plugin(plugins_dir, "weather")
    (dest / "plugin.yaml").write_text(
        "id: weather\nversion: 0.1.0\nname: weather\nkind: hook\nentry: main.py\n",
        encoding="utf-8",
    )
    (dest / "main.py").write_text(
        "from octop_harness.plugins import PluginContext\n"
        "\n"
        "\n"
        "def setup(ctx: PluginContext) -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=tmp_path / "config.json")
    assert mgr.load_installed(install_deps=False, defer_hooks=True) == []
    assert PluginRegistry().get("weather") is None
    loaded = mgr.load_installed(install_deps=False)
    assert [item.manifest.id for item in loaded] == ["weather"]


def test_disable_calls_plugin_teardown(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    dest = _write_plugin(plugins_dir, "worker")
    (dest / "main.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "from octop_harness.plugins import PluginContext\n"
        "\n"
        "\n"
        "def setup(ctx: PluginContext) -> None:\n"
        "    pass\n"
        "\n"
        "\n"
        "def teardown(ctx, reason: str = 'reload') -> None:\n"
        "    Path(__file__).with_name('teardown.txt').write_text(reason, encoding='utf-8')\n",
        encoding="utf-8",
    )
    mgr = PluginManager(plugins_dir=plugins_dir, config_path=tmp_path / "config.json")
    mgr.load_installed(install_deps=False)
    mgr.set_enabled("worker", False)
    assert (dest / "teardown.txt").read_text(encoding="utf-8") == "disable"
