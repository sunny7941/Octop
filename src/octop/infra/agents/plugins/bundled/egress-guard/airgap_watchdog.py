"""Scheduled-task watchdog: enforce the Octop air-gap invariant every minute.

The plugin-spawned helper gives an immediate block at Octop start and an
immediate restore at Octop exit, but it is a plain user process — it can be
killed, and it dies with the session that spawned it. This watchdog is the
safety net: a scheduled task, independent of Octop and of the helper, that
re-establishes the correct firewall state within a couple of minutes no matter
what crashed.

Invariant
---------
* ``airgap_switch.enabled()`` is false  ->  no air-gap rule exists (default).
* Switch on and Octop listening on 127.0.0.1:8088  ->  every air-gap rule present.
* Switch on and Octop not listening for 2 consecutive polls  ->  machine-wide
  rules removed, egress restored.

The firewall work is delegated to ``airgap_helper`` so both mechanisms always
agree on rule names and semantics. Fully local — no network access.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import airgap_helper as h  # noqa: E402
import airgap_switch as sw  # noqa: E402

STATE_FILE = Path(__file__).with_name("watchdog.json")
LOG_PATH = Path(__file__).with_name("watchdog.log")

DOWN_POLLS_REQUIRED = 2  # consecutive "Octop is gone" polls before restoring
ENFORCE_EVERY_TICKS = 5  # re-assert the block at most every N polls while running
PORT_TIMEOUT_SECONDS = 2


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    with contextlib.suppress(OSError):
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def main() -> int:
    state = _load_state()

    # The switch is authoritative: OFF means the machine stays online, so the
    # watchdog only cleans up leftovers and never re-applies the air-gap.
    if not sw.enabled():
        state["running"] = False
        state["down"] = 0
        state["tick"] = 0
        if state.get("switch") != "off":
            _log("air-gap switch is OFF; removing any air-gap rules")
            for line in h.restore():
                _log(f"  {line}")
            state["switch"] = "off"
        _save_state(state)
        return 0
    state["switch"] = "on"

    running = h.port_open()
    previous = state.get("running")
    changed = previous is None or running != previous
    tick = int(state.get("tick", 0)) + 1

    if running:
        state["down"] = 0
        if changed:
            _log("Octop detected (port 8088); applying the air-gap")
            for line in h.ensure_block():
                _log(f"  {line}")
            tick = 0
        elif tick >= ENFORCE_EVERY_TICKS:
            for line in h.ensure_block():
                if not line.endswith(" OK"):
                    _log(f"  {line}")
            tick = 0
        state["tick"] = tick
        state["running"] = True
        _save_state(state)
        return 0

    down = int(state.get("down", 0)) + 1
    state["down"] = down
    state["running"] = False
    state["tick"] = 0
    if down < DOWN_POLLS_REQUIRED and not changed:
        _save_state(state)
        return 0
    _log(f"Octop not seen (poll {down}); restoring egress")
    for line in h.restore():
        _log(f"  {line}")
    state["down"] = 0
    _save_state(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        _log(f"watchdog error: {exc}")
        raise SystemExit(1) from None
