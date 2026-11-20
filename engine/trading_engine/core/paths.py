"""Plugin and workspace paths. The workspace is your repository of data and experiments."""
from __future__ import annotations

import os
from pathlib import Path

from .errors import CatalogError


def plugin_root() -> Path:
    env = os.environ.get("CLAUDE_PLUGIN_ROOT")
    # Without the variable, the package is assumed to run from the plugin tree
    # (engine/trading_engine/core/paths.py), as bin/trading-engine does.
    root = Path(env) if env else Path(__file__).resolve().parents[3]
    if not (root / ".claude-plugin" / "plugin.json").is_file():
        raise CatalogError(
            f"{root} is not the plugin root. Run the engine through `trading-engine` "
            "or set CLAUDE_PLUGIN_ROOT to the folder that contains .claude-plugin/."
        )
    return root


def workspace_root() -> Path:
    env = os.environ.get("TRADING_LAB_HOME")
    return Path(env).expanduser().resolve() if env else Path.cwd()


def catalog_dirs() -> list[Path]:
    """Workspace first (may override), then the plugin (defaults)."""
    return [workspace_root() / "catalog", plugin_root() / "catalog"]
