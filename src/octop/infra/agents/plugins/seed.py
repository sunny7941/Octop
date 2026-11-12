"""Install and upgrade the plugins shipped inside the wheel.

Bundled plugins opt into first-start installation with ``auto_install: true``;
others are only refreshed when already installed. Seed only refreshes a local
copy when the bundled version is newer, and never touches settings stored
outside the plugin directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from octop.infra.errors import corrupt_config_error
from octop.infra.utils.json_file import (
    JsonFileCorruptError,
    read_json_object,
    write_json_atomic,
)


def _read_config(config_path: Path) -> dict[str, Any]:
    """Load ``config.json`` for merging; ``{}`` when absent.

    A corrupt file raises instead of reading as empty: ``seed_bundled_plugins``
    always writes back, so merging into ``{}`` would destroy every other setting
    (issue #730).
    """
    try:
        data = read_json_object(config_path)
    except JsonFileCorruptError as exc:
        raise corrupt_config_error(exc.path, exc.detail) from exc
    return data if data is not None else {}


def _write_config(config_path: Path, data: dict[str, Any]) -> None:
    write_json_atomic(config_path, data)


def _plugin_manifest(plugin_dir: Path) -> dict[str, Any]:
    path = plugin_dir / "plugin.yaml"
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _plugin_version(plugin_dir: Path) -> tuple[int, ...]:
    ver = str(_plugin_manifest(plugin_dir).get("version") or "0")
    parts: list[int] = []
    for bit in ver.split("."):
        try:
            parts.append(int(bit))
        except ValueError:
            parts.append(0)
    return tuple(parts) or (0,)


def plugin_auto_install(plugin_dir: Path) -> bool:
    """Bundled plugins opt into first-start installation with ``auto_install``."""
    return bool(_plugin_manifest(plugin_dir).get("auto_install"))


def plugin_default_enabled(plugin_dir: Path) -> bool:
    """Initial global flag for a bundled plugin (absent means enabled)."""
    return bool(_plugin_manifest(plugin_dir).get("default_enabled", True))


def _apply_default_flag(plugins_cfg: dict[str, Any], plugin_id: str, default: bool) -> None:
    """Set ``plugins.<id>.enabled`` only when the user has no recorded value."""
    entry = plugins_cfg.get(plugin_id)
    if isinstance(entry, dict) and "enabled" in entry:
        return
    merged = dict(entry) if isinstance(entry, dict) else {}
    merged["enabled"] = bool(default)
    plugins_cfg[plugin_id] = merged


def seed_bundled_plugins(
    *,
    bundled_root: Path,
    plugins_dir: Path,
    config_path: Path,
    known_plugins: set[str] | None = None,
) -> list[str]:
    """Install or upgrade bundled plugins.

    - ``auto_install: true`` plugins are copied on first start (ids in
      ``known_plugins`` are skipped so a deliberate uninstall is not undone).
    - Existing dest dirs are overwritten only when the bundled ``plugin.yaml``
      version is newer; the user's enabled flag is preserved.
    - New copies get ``default_enabled`` (from the manifest) when the user has
      no recorded flag yet.
    - Settings stored outside the plugin dir (for example the air-gap switch in
      ``$OCTOP_HOME/airgap``) are never touched.

    Does not write config when nothing changed.
    """
    if not bundled_root.is_dir():
        return []
    known = known_plugins or set()
    data = _read_config(config_path)
    plugins_cfg = data.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        data["plugins"] = plugins_cfg

    copied: list[str] = []
    plugins_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(bundled_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if not (child / "plugin.yaml").is_file():
            continue
        plugin_id = child.name
        dest = plugins_dir / plugin_id
        default_enabled = plugin_default_enabled(child)
        if not dest.exists():
            if not plugin_auto_install(child) or plugin_id in known:
                continue
            shutil.copytree(child, dest)
            copied.append(plugin_id)
            _apply_default_flag(plugins_cfg, plugin_id, default_enabled)
            continue
        if _plugin_version(child) <= _plugin_version(dest):
            continue
        shutil.rmtree(dest)
        shutil.copytree(child, dest)
        copied.append(plugin_id)
        _apply_default_flag(plugins_cfg, plugin_id, default_enabled)

    if not copied:
        return []
    data["plugins"] = plugins_cfg
    _write_config(config_path, data)
    return copied
