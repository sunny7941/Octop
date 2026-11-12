"""In-package plugin directory shipped inside the wheel.

Each subdirectory is one bundled plugin. ``seed_bundled`` upgrades an installed
copy when this catalog has a newer version; plugins that declare
``auto_install: true`` are also installed on first start (disabled when their
manifest says ``default_enabled: false``).
"""

from pathlib import Path


def default_bundled_plugins_root() -> Path:
    """Directory containing one subdirectory per catalog plugin."""
    return Path(__file__).resolve().parent
