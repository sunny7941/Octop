"""Air-gap rule engine shared by the Octop helper and the watchdog.

Two consumers:

* ``airgap_helper.py`` — spawned by the Octop plugin, blocks egress the moment
  the server starts and restores it the moment the server exits.
* ``airgap_watchdog.py`` — a scheduled task that runs every minute and enforces
  the same invariant, so a killed helper, a crashed Octop or a missed restore
  can never leave the machine blocked (or unblocked) for more than a couple of
  minutes.

Rule model
----------
Program-scoped rules (they only match Octop / Ollama binaries):

* Octop Server / Octop Desktop / Ollama Daemon - Block Outbound (air-gap)

Machine-wide rule (no program filter — this is what actually closes the
machine's egress path):

* Air-gap - Block All Outbound (air-gap)

Program paths are resolved on this machine at call time; a program rule whose
binary is not installed is simply skipped (the machine-wide rule still applies).
Cleanup always covers every name, including rules for programs that are absent
or gone.

Switch
------
The air-gap is **off by default**. Every rule is created only while
``airgap_switch.enabled()`` is true; turning the switch off restores egress by
removing all of them. See ``airgap_switch.py`` for the resolution order.

Everything is local — no network access is performed or required.
"""

from __future__ import annotations

import base64
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import airgap_switch

RULE_NAMES: tuple[str, ...] = (
    "Octop Server - Block Outbound (air-gap)",
    "Octop Desktop - Block Outbound (air-gap)",
    "Ollama Daemon - Block Outbound (air-gap)",
    "Air-gap - Block All Outbound (air-gap)",
)

# Rules from earlier designs; cleaned up if they are still present.
LEGACY_RULES: tuple[str, ...] = (
    "Proxy xray - Block Outbound (air-gap)",
    "Proxy mihomo - Block Outbound (air-gap)",
    "Proxy sing-box - Block Outbound (air-gap)",
)

# Every rule this feature owns; removing all of them fully restores egress.
ALL_RULE_NAMES: tuple[str, ...] = RULE_NAMES + LEGACY_RULES

OCTOP_PORT = 8088
POWERSHELL_TIMEOUT = 120

# Helper (plugin-spawned) tuning.
GRACE_SECONDS = 30  # let Octop finish booting before trusting a "down" reading
DOWN_POLLS_REQUIRED = 3  # consecutive "Octop is gone" readings before restoring
POLL_SECONDS = 5

LOG_PATH = Path(__file__).with_name("helper.log")
PIDFILE = Path(__file__).with_name("helper.pid")


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# --- program discovery ------------------------------------------------------


def _first_file(*candidates: str | None) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def _env_path(key: str) -> str | None:
    raw = (os.environ.get(key) or "").strip()
    return raw or None


def _desktop_exe() -> str | None:
    local = os.environ.get("LOCALAPPDATA")
    programs = os.environ.get("PROGRAMFILES")
    return _first_file(
        _env_path("OCTOP_AIRGAP_DESKTOP_EXE"),
        str(Path(local) / "Programs" / "Octop" / "Octop.exe") if local else None,
        str(Path(programs) / "Octop" / "Octop.exe") if programs else None,
    )


def _ollama_exe() -> str | None:
    local = os.environ.get("LOCALAPPDATA")
    return _first_file(
        _env_path("OCTOP_AIRGAP_OLLAMA_EXE"),
        shutil.which("ollama"),
        str(Path(local) / "Programs" / "Ollama" / "ollama.exe") if local else None,
    )


def airgap_rules() -> tuple[tuple[str, str | None], ...]:
    """Rules to create, with program paths resolved on this machine."""
    entries: list[tuple[str, str | None]] = []
    program_rules = (
        (RULE_NAMES[0], str(Path(sys.executable)) if sys.executable else None),
        (RULE_NAMES[1], _desktop_exe()),
        (RULE_NAMES[2], _ollama_exe()),
    )
    for name, path in program_rules:
        if path:
            entries.append((name, path))
    # Blanket rule: no -Program, so it applies to every application. This is the
    # only thing that stops direct (non-proxy) egress.
    entries.append((RULE_NAMES[3], None))
    return tuple(entries)


# --- PowerShell plumbing ----------------------------------------------------


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
    # CREATE_NO_WINDOW: the caller (watchdog / helper) runs under pythonw with no
    # console, so a bare console child would make Windows pop a temporary
    # terminal window on every firewall sync. getattr keeps non-Windows ports happy.
    # Octop forces Python into UTF-8 mode while PowerShell emits the OEM code
    # page, so pin the child to UTF-8 and decode leniently.
    script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n" + script
    proc = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-EncodedCommand", _encode(script)],
        capture_output=True,
        timeout=POWERSHELL_TIMEOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def _decode(raw: bytes | None) -> str:
        return (raw or b"").decode("utf-8", errors="replace").strip()

    return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)


def _pair_lines(entries: list[tuple[str, str | None]]) -> list[str]:
    lines = []
    for name, path in entries:
        if path is None:
            lines.append("  @{ Name = " + _q(name) + "; Path = $null }")
        else:
            lines.append("  @{ Name = " + _q(name) + "; Path = " + _q(path) + " }")
    return lines


def _ensure_script(entries: list[tuple[str, str | None]]) -> str:
    return "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            "$ErrorActionPreference = 'Stop'",
            "$pairs = @(",
            ",\n".join(_pair_lines(entries)),
            ")",
            "foreach ($p in $pairs) {",
            "  try {",
            "    $existing = @(Get-NetFirewallRule -DisplayName $p.Name -ErrorAction SilentlyContinue)",
            "    $ok = $false",
            "    if ($existing.Count -gt 0) {",
            "      $ok = ([string]$existing[0].Enabled -eq 'True') -and ([string]$existing[0].Action -eq 'Block')",
            "      if ($ok -and $p.Path -ne $null) {",
            "        $app = @($existing[0] | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue)",
            "        $prog = if ($app.Count -gt 0) { [string]$app[0].Program } else { '' }",
            "        $ok = $prog -eq $p.Path",
            "      }",
            "    }",
            "    if (-not $ok) {",
            "      $verb = 'CREATED'",
            "      if ($existing.Count -gt 0) { Remove-NetFirewallRule -DisplayName $p.Name -ErrorAction Stop; $verb = 'FIXED' }",
            "      if ($p.Path -eq $null) {",
            "        New-NetFirewallRule -DisplayName $p.Name -Direction Outbound -Action Block -Profile Any -ErrorAction Stop | Out-Null",
            "      } else {",
            "        New-NetFirewallRule -DisplayName $p.Name -Direction Outbound -Action Block -Program $p.Path -Profile Any -ErrorAction Stop | Out-Null",
            "      }",
            "      Write-Output ($p.Name + ' ' + $verb)",
            "    } else {",
            "      Write-Output ($p.Name + ' OK')",
            "    }",
            "  } catch {",
            "    Write-Output ($p.Name + ' ERROR ' + $_.Exception.Message)",
            "  }",
            "}",
        ]
    )


def _check_script(entries: list[tuple[str, str | None]]) -> str:
    return "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            "$ErrorActionPreference = 'Continue'",
            "$pairs = @(",
            ",\n".join(_pair_lines(entries)),
            ")",
            "foreach ($p in $pairs) {",
            "  try {",
            "    $r = @(Get-NetFirewallRule -DisplayName $p.Name -ErrorAction SilentlyContinue)",
            "    if ($r.Count -eq 0) { Write-Output ($p.Name + ' => MISSING'); continue }",
            "    if ([string]$r[0].Enabled -ne 'True') { Write-Output ($p.Name + ' => DISABLED'); continue }",
            "    if ([string]$r[0].Action -ne 'Block') { Write-Output ($p.Name + ' => ACTION ' + $r[0].Action); continue }",
            "    if ($p.Path -ne $null) {",
            "      $app = @($r[0] | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue)",
            "      $prog = if ($app.Count -gt 0) { [string]$app[0].Program } else { '' }",
            "      if ($prog -ne $p.Path) { Write-Output ($p.Name + ' => WRONG_PATH ' + $prog); continue }",
            "    }",
            "    Write-Output ($p.Name + ' => OK')",
            "  } catch {",
            "    Write-Output ($p.Name + ' => ERROR ' + $_.Exception.Message)",
            "  }",
            "}",
        ]
    )


def _remove_script(names: list[str]) -> str:
    return "\n".join(
        [
            "$ProgressPreference = 'SilentlyContinue'",
            "$ErrorActionPreference = 'Stop'",
            "$names = @(",
            ",\n".join("  " + _q(name) for name in names),
            ")",
            "foreach ($n in $names) {",
            "  try {",
            "    $existing = @(Get-NetFirewallRule -DisplayName $n -ErrorAction SilentlyContinue)",
            "    if ($existing.Count -gt 0) {",
            "      Remove-NetFirewallRule -DisplayName $n -ErrorAction Stop",
            "      Write-Output ($n + ' REMOVED')",
            "    } else {",
            "      Write-Output ($n + ' ABSENT')",
            "    }",
            "  } catch {",
            "    Write-Output ($n + ' ERROR ' + $_.Exception.Message)",
            "  }",
            "}",
        ]
    )


# --- operations -------------------------------------------------------------


def ensure_block() -> list[str]:
    """Create/fix every air-gap rule. Returns the per-rule result lines."""
    code, out, err = _run_ps(_ensure_script(list(airgap_rules())))
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if err:
        lines.append(f"stderr: {err}")
    if code != 0 and not lines:
        lines.append(f"powershell exited {code}")
    return lines


def restore() -> list[str]:
    """Remove every air-gap rule so egress works again."""
    names = list(ALL_RULE_NAMES)
    code, out, err = _run_ps(_remove_script(names))
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if err:
        lines.append(f"stderr: {err}")
    if code != 0 and not lines:
        lines.append(f"powershell exited {code}")
    return lines


def report() -> list[str]:
    code, out, err = _run_ps(_check_script(list(airgap_rules())))
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if err:
        lines.append(f"stderr: {err}")
    return lines


def port_open() -> bool:
    """True when something accepts a TCP connection on the Octop port."""
    try:
        with socket.create_connection(("127.0.0.1", OCTOP_PORT), timeout=2):
            return True
    except OSError:
        return False


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes
    except ImportError:  # pragma: no cover
        return False
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _pidfile_alive() -> bool:
    try:
        pid = int(PIDFILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return pid_alive(pid)


def main() -> int:
    """Plugin entry point: block now, watch Octop, restore on exit."""
    if "--restore" in sys.argv[1:]:
        _log("one-shot restore requested; removing every air-gap rule")
        for line in restore():
            _log(f"  {line}")
        _log("restored; exiting")
        return 0

    octop_pid = 0
    if len(sys.argv) > 1:
        try:
            octop_pid = int(sys.argv[1])
        except ValueError:
            octop_pid = 0
    if octop_pid <= 0:
        _log("no Octop pid supplied; aborting")
        return 2
    if _pidfile_alive():
        _log("another helper is already running; exiting")
        return 0
    if not airgap_switch.enabled():
        _log("air-gap switch is OFF; restoring egress and exiting")
        for line in restore():
            _log(f"  {line}")
        return 0

    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    _log(f"helper started pid={os.getpid()} octop_pid={octop_pid}")
    try:
        for line in ensure_block():
            _log(f"  {line}")

        deadline = time.monotonic() + GRACE_SECONDS
        down_streak = 0
        while True:
            time.sleep(POLL_SECONDS)
            if not airgap_switch.enabled():
                _log("air-gap switch turned OFF; restoring egress")
                break
            alive = pid_alive(octop_pid) or port_open()
            if alive:
                down_streak = 0
                continue
            down_streak += 1
            if time.monotonic() < deadline:
                continue
            _log(
                f"Octop not seen (pid={pid_alive(octop_pid)} port={port_open()}) "
                f"for {down_streak} poll(s)"
            )
            if down_streak >= DOWN_POLLS_REQUIRED:
                break

        _log("restoring egress")
        for line in restore():
            _log(f"  {line}")
        _log("restored; helper exiting")
        return 0
    finally:
        with contextlib.suppress(OSError):
            PIDFILE.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
