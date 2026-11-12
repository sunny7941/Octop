"""Octop lifetime air-gap: block egress while Octop runs, restore when it exits.

The air-gap is **off by default**. Only when the user explicitly turns the
switch on (``airgap_switch.py on``, the ``switch.json`` file, or the environment
variable ``OCTOP_AIRGAP_ENABLED=1``) does this plugin start the elevated helper.

The plugin ships its runtime scripts next to ``main.py`` and copies them into
``$OCTOP_HOME/airgap`` (default ``~/.octop/airgap``) on load, so the watchdog
scheduled task and the CLI entry point keep a stable path. Existing
``switch.json`` state is never touched.

The plugin's only job is to start the elevated helper exactly once per Octop
server lifetime. The helper (``airgap_helper.py``) owns the actual firewall
work, because Octop's plugin system has no shutdown/teardown hook — only an
independent process can reliably restore egress when the server is closed or
crashes.

    Octop start -> switch OFF? -> stay idle (machine stays online)
                -> switch ON  -> one UAC prompt -> helper
                                              |- ensure every Block rule exists
                                              |- poll Octop pid + port 8088
                                              |- Octop gone or switch off ->
                                                 remove every rule, exit

One UAC prompt per Octop session; restoration is fully automatic. The helper is
crash-safe: it is an independent process, so killing Octop cannot leave the
machine-wide rules behind.

Windows-only and fully local — the plugin performs no network access.
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPTS_VERSION = "1.4.1"
_PLUGIN_DIR = Path(__file__).resolve().parent
_OCTOP_HOME = Path(
    os.environ.get("OCTOP_HOME") or Path(os.environ.get("USERPROFILE") or Path.home()) / ".octop"
)
_HELPER_DIR = _OCTOP_HOME / "airgap"
_RUNTIME_SCRIPTS = (
    "airgap_switch.py",
    "airgap_helper.py",
    "airgap_watchdog.py",
    "install_watchdog.ps1",
)
_SCRIPTS_VERSION_FILE = _HELPER_DIR / "scripts.version"
_HELPER = _HELPER_DIR / "airgap_helper.py"
_PIDFILE = _HELPER_DIR / "helper.pid"
_HELPER_PYTHON = Path(sys.executable).with_name("pythonw.exe")
if not _HELPER_PYTHON.is_file():
    _HELPER_PYTHON = Path(sys.executable)
_THREAD_NAME = "octop-egress-guard"
_GEN_FILE = Path(__file__).with_name(".egress-guard-gen")
_ELEVATE_TIMEOUT_SECONDS = 300
_POLL_SECONDS = 5  # settings changes take effect within a few seconds
_SPAWN_RETRY_SECONDS = 120  # at most one UAC prompt per two minutes when declined
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

airgap_switch = None
_stop_event = threading.Event()
_stop_lock = threading.Lock()


def _sync_runtime_scripts() -> None:
    """Copy this plugin version's scripts into ``$OCTOP_HOME/airgap``."""
    try:
        installed = _SCRIPTS_VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        installed = ""
    if installed == _SCRIPTS_VERSION:
        return
    _HELPER_DIR.mkdir(parents=True, exist_ok=True)
    for name in _RUNTIME_SCRIPTS:
        source = _PLUGIN_DIR / name
        if source.is_file():
            shutil.copy2(source, _HELPER_DIR / name)
    _SCRIPTS_VERSION_FILE.write_text(_SCRIPTS_VERSION + "\n", encoding="utf-8")
    logger.info("[egress-guard] seeded runtime scripts into %s", _HELPER_DIR)


def _init_switch() -> None:
    """Seed the scripts, then import the switch; stays ``None`` on failure."""
    global airgap_switch
    try:
        _sync_runtime_scripts()
    except OSError as exc:
        logger.warning("[egress-guard] cannot seed runtime scripts: %s", exc)
    sys.path.insert(0, str(_HELPER_DIR))
    try:
        import airgap_switch as module  # noqa: PLC0415 - after the seed copy
    except Exception as exc:  # noqa: BLE001 - never break Octop startup
        logger.warning("[egress-guard] air-gap switch unavailable: %s", exc)
        airgap_switch = None
        return
    airgap_switch = module


def _setting_enabled() -> bool:
    """True only when the air-gap switch is explicitly on (default: off)."""
    if airgap_switch is None:
        return False
    try:
        return airgap_switch.enabled()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[egress-guard] switch lookup failed: %s", exc)
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _pid_image(pid: int) -> str | None:
    """Executable path of *pid*, or None when it cannot be queried."""
    if pid <= 0:
        return None
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = ctypes.c_uint(260)
        buf = ctypes.create_unicode_buffer(260)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def _helper_running() -> bool:
    """True when the pid file names a live process that is really the helper."""
    try:
        pid = int(_PIDFILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if not _pid_alive(pid):
        return False
    image = _pid_image(pid)
    if image is None:
        return False
    expected = str(_HELPER_PYTHON).lower()
    return image.lower() == expected or Path(image).name.lower() in {"python.exe", "pythonw.exe"}


def _powershell() -> str:
    found = shutil.which("powershell")
    if found:
        return found
    fallback = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("powershell.exe not found")


def _encode(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _q(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_ps(script: str) -> tuple[int, str, str]:
    script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n" + script
    proc = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-EncodedCommand", _encode(script)],
        capture_output=True,
        timeout=_ELEVATE_TIMEOUT_SECONDS,
    )

    def _decode(raw: bytes | None) -> str:
        return (raw or b"").decode("utf-8", errors="replace").strip()

    return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)


def _spawn_helper() -> bool:
    """Start the helper elevated. Returns True when it exited 0."""
    if not _HELPER.is_file():
        logger.error("[egress-guard] helper not found at %s", _HELPER)
        return False
    if _helper_running():
        logger.info("[egress-guard] helper already running; not spawning a second one")
        return True

    script = "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            "$ErrorActionPreference = 'Stop'",
            "try {",
            f"  Start-Process -Verb RunAs -FilePath {_q(str(_HELPER_PYTHON))} "
            f"-ArgumentList {_q(str(_HELPER))},{_q(str(os.getpid()))} | Out-Null",
            "  Write-Output 'STARTED'",
            "} catch {",
            "  Write-Output ('ERROR ' + $_.Exception.Message)",
            "  exit 1",
            "}",
        ]
    )
    logger.info("[egress-guard] starting elevated helper (one UAC prompt)")
    code, out, err = _run_ps(script)
    for line in out.splitlines():
        if line.strip():
            logger.info("[egress-guard] helper: %s", line.strip())
    if err:
        logger.warning("[egress-guard] helper stderr: %s", err)
    # Deliberately fire-and-forget: the helper lives for the whole Octop
    # session, so waiting for it would block the monitoring loop until Octop
    # exits. The pid file is the liveness signal instead.
    return code == 0


def _current_generation() -> str:
    try:
        return _GEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _new_stop_event() -> threading.Event:
    """Replace the process-wide stop signal before starting a new monitor."""
    global _stop_event
    with _stop_lock:
        _stop_event = threading.Event()
        return _stop_event


def _request_stop() -> None:
    with _stop_lock:
        _stop_event.set()


def _stop_monitor_threads(timeout: float = 3.0) -> None:
    """Signal the monitor thread(s) and give them a moment to exit."""
    _request_stop()
    current = threading.current_thread()
    for thread in threading.enumerate():
        if thread is current or thread.name != _THREAD_NAME:
            continue
        thread.join(timeout)


def _loop(generation: str, stop: threading.Event) -> None:
    last_state: bool | None = None
    next_spawn_at = 0.0
    while not stop.is_set():
        if _current_generation() != generation:
            logger.info("[egress-guard] superseded by a newer load; exiting")
            return
        try:
            on = _setting_enabled()
            if on != last_state:
                logger.info("[egress-guard] air-gap switch is %s", "ON" if on else "OFF (idle)")
                last_state = on
            if not on:
                # Keep polling so turning the switch on later in the same Octop
                # session still takes effect; the helper restores egress itself
                # when it sees the switch flip off.
                stop.wait(_POLL_SECONDS)
                continue
            # The helper is the only thing that can restore egress, so a dead
            # helper is re-spawned (one UAC prompt each time it actually dies).
            now = time.monotonic()
            if not _helper_running() and now >= next_spawn_at:
                _spawn_helper()
                next_spawn_at = now + _SPAWN_RETRY_SECONDS
        except Exception as exc:  # noqa: BLE001
            logger.warning("[egress-guard] check failed: %s", exc)
        stop.wait(_POLL_SECONDS)


def _helper_pid() -> int:
    try:
        return int(_PIDFILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _kill_helper(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("[egress-guard] cannot stop helper pid=%s: %s", pid, exc)


def _stop_helper(grace: float = 7.0) -> None:
    """Give the helper a moment to notice the switch is off, then stop it.

    The helper polls the switch every 5 seconds and restores egress by itself;
    killing it first would throw that away (and the server process usually
    cannot delete firewall rules without elevation). *grace* covers one poll
    plus slack; only a helper still alive after that is force-stopped.
    """
    pid = _helper_pid()
    if pid <= 0 or not _pid_alive(pid):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    logger.info("[egress-guard] helper pid=%s did not exit; stopping it", pid)
    _kill_helper(pid)


def _remove_rules() -> None:
    """Remove every air-gap rule (current and legacy), if allowed."""
    try:
        import airgap_helper  # noqa: PLC0415 - seeded next to this plugin

        for line in airgap_helper.restore():
            logger.info("[egress-guard] cleanup: %s", line)
    except Exception as exc:  # noqa: BLE001 - cleanup must never raise
        logger.warning("[egress-guard] rule cleanup failed: %s", exc)


def _cleanup() -> None:
    """Turn the switch off, stop the helper and drop every air-gap rule.

    Idempotent: every step tolerates missing settings rows, a dead helper and
    absent rules. The scheduled watchdog reads the same switch, so once it is
    off the task only cleans up and never re-blocks the machine.
    """
    if airgap_switch is None:
        _init_switch()
    if airgap_switch is None:
        logger.warning("[egress-guard] switch unavailable; cannot persist OFF")
    else:
        try:
            source = airgap_switch.source()
            if source.startswith("override"):
                logger.warning(
                    "[egress-guard] %s still wins over the setting; run "
                    "airgap_switch.py clear-override to restore egress",
                    source,
                )
            elif source.startswith("env"):
                logger.warning(
                    "[egress-guard] %s still wins over the setting; unset it to restore egress",
                    source,
                )
            airgap_switch.set_enabled(False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[egress-guard] cannot persist switch OFF: %s", exc)
    _stop_helper()
    _remove_rules()
    if not _pid_alive(_helper_pid()):
        with contextlib.suppress(OSError):
            _PIDFILE.unlink()


def teardown(ctx=None, reason: str = "reload") -> None:
    """Octop calls this before a plugin is disabled, reloaded or uninstalled.

    Always stop the monitor thread so it can no longer spawn the helper.
    ``disable`` / ``uninstall`` additionally restore egress and persist OFF.
    """
    _stop_monitor_threads()
    logger.info("[egress-guard] monitor stopped (%s)", reason)
    if reason in {"disable", "uninstall"}:
        _cleanup()
        logger.info("[egress-guard] cleanup finished (%s)", reason)


def setup(ctx) -> None:
    if os.name != "nt":
        logger.info("[egress-guard] Windows-only plugin; staying idle")
        return
    _init_switch()
    _stop_monitor_threads()
    stop = _new_stop_event()
    generation = f"{time.time():.6f}"
    try:
        _GEN_FILE.write_text(generation, encoding="utf-8")
    except OSError as exc:
        logger.warning("[egress-guard] cannot write generation file: %s", exc)
    worker = threading.Thread(
        target=_loop,
        args=(generation, stop),
        name=_THREAD_NAME,
        daemon=True,
    )
    worker._octop_generation = generation  # type: ignore[attr-defined]
    worker.start()
    logger.info("[egress-guard] started")
