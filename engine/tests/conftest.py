from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Each test sees only the plugin catalog and an empty workspace, independent of the user's environment."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(PLUGIN_ROOT))
    monkeypatch.setenv("TRADING_LAB_HOME", str(workspace))
    return workspace


@pytest.fixture
def binance_asset(isolated_env):
    """A synthetic asset on the Binance USD-M source, declared in the workspace like a real one."""
    (isolated_env / "catalog").mkdir(exist_ok=True)
    (isolated_env / "catalog" / "assets.yaml").write_text(
        "XYZUSDT:\n  market: crypto_perp_usdt\n  source: binance_um_futures\n  streams: [trades]\n", encoding="utf-8"
    )
    return "XYZUSDT"
