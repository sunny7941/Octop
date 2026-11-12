"""Single source of truth for the Octop air-gap switch.

Default is **off**: Octop must never cut the machine's network unless the user
explicitly turns the air-gap on.

Resolution order (first match wins):

1. Environment variable ``OCTOP_AIRGAP_ENABLED`` (1/true/yes/on or 0/false/no/off)
2. ``$OCTOP_HOME/airgap/switch.json`` — explicit *emergency override*
3. Octop ``settings`` row ``airgap_enabled`` — the product setting shown at
   *Settings → App settings → Air-gap*
4. ``False``

``on`` / ``off`` write the product setting, so the Settings page and this CLI
always operate the same state. ``switch.json`` is written only with an explicit
``--override`` flag, for emergencies where the server/UI is unreachable; it then
wins over the Settings page and has to be cleared explicitly.

Usage::

    python airgap_switch.py status            # switch, its source, firewall rules
    python airgap_switch.py on                # save ON to the product settings
    python airgap_switch.py off               # save OFF and restore egress
    python airgap_switch.py on --override     # emergency override ON (switch.json)
    python airgap_switch.py off --override    # emergency override OFF (switch.json)
    python airgap_switch.py clear-override    # delete switch.json

Only ``off`` may need elevation (removing firewall rules); the scheduled-task
watchdog re-checks the switch every minute and cleans up if that is skipped.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OCTOP_HOME = Path(os.environ.get("OCTOP_HOME") or HERE.parent)
CONFIG_FILE = OCTOP_HOME / "config.json"
SWITCH_FILE = OCTOP_HOME / "airgap" / "switch.json"
ENV_KEY = "OCTOP_AIRGAP_ENABLED"
DB_SETTING_KEY = "airgap_enabled"
HELPER = HERE / "airgap_helper.py"
HELPER_PIDFILE = HERE / "helper.pid"


class SwitchConfigError(RuntimeError):
    """User-facing failure while reading or updating the switch."""


def _script_name() -> str:
    return Path(sys.argv[0]).name if sys.argv else "airgap_switch.py"


RULE_NAMES: tuple[str, ...] = (
    "Octop Server - Block Outbound (air-gap)",
    "Octop Desktop - Block Outbound (air-gap)",
    "Ollama Daemon - Block Outbound (air-gap)",
    "Air-gap - Block All Outbound (air-gap)",
)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_value() -> bool | None:
    raw = os.environ.get(ENV_KEY)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def _file_value() -> bool | None:
    try:
        data = json.loads(SWITCH_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and "enabled" in data:
        return bool(data["enabled"])
    return None


def _database_config() -> tuple[str, Path]:
    """Resolve Octop's DB driver and SQLite path (env > config.json > defaults)."""
    driver = (os.environ.get("OCTOP_DATABASE_DRIVER") or "").strip().lower()
    sqlite_path = (os.environ.get("OCTOP_DATABASE_SQLITE_PATH") or "").strip()
    if not driver and os.environ.get("OCTOP_DATABASE_URL"):
        driver = "postgresql"
    if not driver or not sqlite_path:
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        database = data.get("database") if isinstance(data, dict) else None
        if isinstance(database, dict):
            driver = driver or str(database.get("driver") or "sqlite").strip().lower()
            sqlite_path = sqlite_path or str(database.get("sqlite_path") or "").strip()
    path = Path(sqlite_path or "octop.db")
    return driver or "sqlite", path if path.is_absolute() else OCTOP_HOME / path


def _db_value() -> bool | None:
    driver, db = _database_config()
    if driver != "sqlite":
        return None
    try:
        if not db.is_file():
            return None
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (DB_SETTING_KEY,)
            ).fetchone()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - a locked DB must not flip the switch
        return None
    if row is None:
        return None
    return str(row[0]).strip().lower() in _TRUE


def enabled() -> bool:
    """True only when the air-gap has been explicitly enabled."""
    for value in (_env_value(), _file_value(), _db_value()):
        if value is not None:
            return value
    return False


def source() -> str:
    """Human-readable description of where the current value comes from."""
    if _env_value() is not None:
        return f"env {ENV_KEY}"
    if _file_value() is not None:
        return f"override {SWITCH_FILE}"
    driver, db = _database_config()
    if _db_value() is not None:
        return f"settings.airgap_enabled ({db})"
    if driver != "sqlite":
        return f"default (database.driver={driver}; settings unreadable)"
    return "default"


def set_enabled(flag: bool) -> Path:
    """Write the product setting shown on the Octop Settings page."""
    driver, db = _database_config()
    if driver != "sqlite":
        raise SwitchConfigError(
            f"Octop uses database.driver={driver}; this CLI can only edit SQLite settings. "
            "Turn the switch off on the Octop Settings page instead."
        )
    if not db.is_file():
        raise SwitchConfigError(
            f"Octop database not found at {db}. Start Octop once (or set OCTOP_HOME) so the "
            "settings row exists, or use --override for a temporary emergency override."
        )
    try:
        conn = sqlite3.connect(db, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            has_settings = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'settings'"
            ).fetchone()
            if has_settings is None:
                raise SwitchConfigError(
                    f"{db} has no settings table yet; start Octop once so migrations run."
                )
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (DB_SETTING_KEY, "true" if flag else "false"),
            )
            conn.commit()
        finally:
            conn.close()
    except SwitchConfigError:
        raise
    except sqlite3.Error as exc:
        raise SwitchConfigError(f"cannot update {db}: {exc}") from exc
    return db


def set_override(flag: bool) -> Path:
    """Write the emergency override file. Raises when the folder is not writable."""
    try:
        SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        SWITCH_FILE.write_text(
            json.dumps({"enabled": bool(flag)}, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise SwitchConfigError(
            f"cannot write override file {SWITCH_FILE}: {exc}. Run this from an account that "
            "can write to that folder, or set the switch on the Octop Settings page."
        ) from exc
    return SWITCH_FILE


def clear_override() -> bool:
    """Delete the emergency override file. Returns False when it was absent."""
    try:
        SWITCH_FILE.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SwitchConfigError(f"cannot remove override file {SWITCH_FILE}: {exc}") from exc
    return True


def _settings_summary() -> str:
    driver, db = _database_config()
    if driver != "sqlite":
        return f"not available (database.driver={driver}; use the Settings page)"
    value = _db_value()
    if value is not None:
        return f"{'ON' if value else 'OFF'} ({db})"
    if not db.is_file():
        return f"not set (database not found: {db})"
    return f"not set or unreadable ({db})"


def _powershell() -> str:
    found = shutil.which("powershell")
    if found:
        return found
    fallback = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("powershell.exe not found")


def _run_ps(script: str) -> tuple[int, str, str]:
    script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n" + script
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    proc = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def _decode(raw: bytes | None) -> str:
        return (raw or b"").decode("utf-8", errors="replace").strip()

    return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)


def rules_present() -> bool | None:
    """Best-effort read-only probe: True/False, or None when unreadable."""
    names = ",\n".join("  '" + name.replace("'", "''") + "'" for name in RULE_NAMES)
    script = "\n".join(
        [
            "$ErrorActionPreference = 'SilentlyContinue'",
            "$names = @(",
            names,
            ")",
            "$count = 0",
            "foreach ($n in $names) {",
            "  if (@(Get-NetFirewallRule -DisplayName $n -ErrorAction SilentlyContinue).Count -gt 0) { $count++ }",
            "}",
            "Write-Output $count",
        ]
    )
    code, out, _err = _run_ps(script)
    if code != 0 or not out.strip().isdigit():
        return None
    return int(out.strip()) > 0


def _helper_pid_alive() -> bool:
    try:
        pid = int(HELPER_PIDFILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _spawn_restore() -> bool:
    """Run one elevated restore (one UAC prompt). Returns True when started."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.is_file() else Path(sys.executable)
    quoted = lambda value: "'" + str(value).replace("'", "''") + "'"  # noqa: E731
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "try {",
            f"  Start-Process -Verb RunAs -FilePath {quoted(exe)} "
            f"-ArgumentList {quoted(HELPER)},{quoted('--restore')} | Out-Null",
            "  Write-Output 'STARTED'",
            "} catch {",
            "  Write-Output ('ERROR ' + $_.Exception.Message)",
            "  exit 1",
            "}",
        ]
    )
    code, out, _err = _run_ps(script)
    print(out.strip())
    return code == 0


def _warn_if_masked(wanted: str) -> None:
    """Point out when env/override keeps the effective value from matching."""
    current = "ON" if enabled() else "OFF"
    if current == wanted:
        return
    print(f"warning: effective state is still {current} (source: {source()}).")
    print("         the env var or emergency override wins over the saved setting.")
    print(f"         clear it with: {_script_name()} clear-override")


def _cmd_status() -> int:
    state = "ON" if enabled() else "OFF"
    print(f"air-gap switch: {state} (source: {source()})")
    env = _env_value()
    if env is not None:
        print(f"  env {ENV_KEY} = {'ON' if env else 'OFF'} (highest priority)")
    file_value = _file_value()
    if file_value is not None:
        print(f"  override {SWITCH_FILE} = {'ON' if file_value else 'OFF'} (wins over settings)")
        print(f"    clear with: {_script_name()} clear-override")
    elif SWITCH_FILE.is_file():
        print(f"  override {SWITCH_FILE} exists but is unreadable; ignored")
    print(f"  settings.airgap_enabled = {_settings_summary()}")
    present = rules_present()
    if present is None:
        print("firewall rules: unknown (requires elevation to inspect)")
    else:
        print(f"firewall rules: {'present' if present else 'absent'}")
    return 0


def _cmd_on(override: bool) -> int:
    if override:
        path = set_override(True)
        print(f"air-gap switch: ON (EMERGENCY OVERRIDE written to {path})")
        print("This file wins over the Octop Settings page, so the UI cannot turn the")
        print("air-gap off until the override is cleared.")
        print(f"Clear it with: {_script_name()} clear-override")
    else:
        db = set_enabled(True)
        print(f"air-gap switch: ON (saved to Octop settings: {db})")
        print("The Settings page shows the same value; Octop applies the block within")
        print("~2 minutes (plugin/watchdog) if it is running, or at the next start.")
    _warn_if_masked("ON")
    return 0


def _cmd_off(override: bool) -> int:
    if override:
        path = set_override(False)
        print(f"air-gap switch: OFF (EMERGENCY OVERRIDE written to {path})")
        print("Delete the file to fall back to the value saved on the Settings page.")
        print(f"Clear it with: {_script_name()} clear-override")
    else:
        db = set_enabled(False)
        print(f"air-gap switch: OFF (saved to Octop settings: {db})")
    if enabled():
        print(f"warning: effective state is still ON (source: {source()}).")
        print("         clear the emergency override (clear-override) or the env var;")
        print("         the watchdog keeps the block while the switch reads ON.")
        return 1
    if _helper_pid_alive():
        print("helper is running; it will remove every air-gap rule within ~10 seconds.")
        return 0
    present = rules_present()
    if present is False:
        print("no air-gap firewall rules to remove.")
        return 0
    if present is None:
        print("cannot inspect firewall rules without elevation; the OctopAirGap")
        print("watchdog will restore egress within ~1 minute if any rule remains.")
        return 0
    print("removing firewall rules now (one UAC prompt)...")
    if _spawn_restore():
        return 0
    print("elevation was declined; the OctopAirGap watchdog will restore egress within ~1 minute.")
    return 0


def _cmd_clear_override() -> int:
    removed = clear_override()
    if removed:
        print(f"removed override {SWITCH_FILE}")
    else:
        print(f"no override file at {SWITCH_FILE}; nothing to clear")
    print(f"air-gap switch: {'ON' if enabled() else 'OFF'} (source: {source()})")
    return 0


def main(argv: list[str]) -> int:
    args = [arg for arg in argv[1:] if arg != "--override"]
    override = "--override" in argv[1:]
    action = args[0].strip().lower() if args else "status"
    try:
        if action == "on":
            return _cmd_on(override)
        if action == "off":
            return _cmd_off(override)
        if action in {"status", "show", ""}:
            return _cmd_status()
        if action in {"clear-override", "clear_override"}:
            return _cmd_clear_override()
    except SwitchConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"usage: {Path(argv[0]).name} [status|on|off [--override]|clear-override]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
