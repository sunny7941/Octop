"""Air-gap switch shared by the dashboard settings API and the egress-guard plugin.

The switch is off by default: Octop never touches the machine's network unless
the user turns it on from *Settings → App settings → Air-gap*.

Resolution order (first match wins), matching the plugin's runtime scripts:

1. Environment variable ``OCTOP_AIRGAP_ENABLED`` (internal override)
2. ``$OCTOP_HOME/airgap/switch.json`` (internal override)
3. ``settings`` row ``airgap_enabled`` — the value the dashboard writes
4. ``False``

``active`` is a best-effort local signal (the helper pid file) telling the
dashboard that the elevated helper is running and the rules are applied.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from octop.infra.db.repos.settings import SettingsRepo

SETTING_KEY = "airgap_enabled"
ENV_KEY = "OCTOP_AIRGAP_ENABLED"
AIRGAP_DIR = "airgap"

AirGapSource = Literal["env", "switch_file", "settings", "default"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class AirGapState:
    enabled: bool
    source: AirGapSource
    active: bool
    supported: bool


def supported() -> bool:
    """The egress controls are Windows firewall rules."""
    return os.name == "nt"


def switch_file(root: Path) -> Path:
    return root / AIRGAP_DIR / "switch.json"


def _parse(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _env_value() -> bool | None:
    return _parse(os.environ.get(ENV_KEY))


def _file_value(root: Path) -> bool | None:
    try:
        data = json.loads(switch_file(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and "enabled" in data:
        return bool(data["enabled"])
    return None


def _settings_value(settings_repo: SettingsRepo) -> bool | None:
    try:
        return _parse(settings_repo.get(SETTING_KEY))
    except Exception:  # noqa: BLE001 - a DB hiccup must not arm the air-gap
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or not supported():
        return False
    # ``ctypes.windll`` only exists on Windows; this function already returned
    # above on other platforms, and ``getattr`` keeps mypy happy on both.
    kernel32 = getattr(ctypes, "windll").kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _helper_active(root: Path) -> bool:
    """True when the elevated helper's pid file points at a live process."""
    try:
        pid = int((root / AIRGAP_DIR / "helper.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def resolve(settings_repo: SettingsRepo, root: Path) -> AirGapState:
    enabled = False
    source: AirGapSource = "default"
    for value, origin in (
        (_env_value(), "env"),
        (_file_value(root), "switch_file"),
        (_settings_value(settings_repo), "settings"),
    ):
        if value is not None:
            enabled = value
            source = origin  # type: ignore[assignment]
            break
    return AirGapState(
        enabled=enabled,
        source=source,
        active=enabled and _helper_active(root),
        supported=supported(),
    )


def set_enabled(settings_repo: SettingsRepo, enabled: bool) -> None:
    """Persist the dashboard switch. The plugin/helper/watchdog poll this row."""
    settings_repo.set(SETTING_KEY, "true" if enabled else "false")
