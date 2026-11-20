import pytest

from trading_engine import catalog
from trading_engine.core.errors import CatalogError, MissingCapabilityError, UnimplementedError


def test_plugin_declares_no_assets():
    assert catalog.load_assets() == {}
    with pytest.raises(CatalogError, match="not declared"):
        catalog.resolve("ANYTHING")


def test_quotes_blocked_for_binance_source(binance_asset):
    with pytest.raises(MissingCapabilityError):
        catalog.resolve(binance_asset, requires=["quotes"])


def test_unimplemented_market_refuses(isolated_env):
    (isolated_env / "catalog").mkdir(exist_ok=True)
    (isolated_env / "catalog" / "assets.yaml").write_text(
        "AAPL:\n  market: equity_us\n  source: generic_csv\n  streams: [trades]\n", encoding="utf-8"
    )
    with pytest.raises(UnimplementedError):
        catalog.resolve("AAPL")


def test_workspace_catalog_merges_by_id(isolated_env):
    (isolated_env / "catalog").mkdir()
    (isolated_env / "catalog" / "sources.yaml").write_text(
        "my_csv:\n  status: implemented\n  capabilities:\n"
        "    trades: true\n    quotes: false\n    aggressor_side: false\n"
        "    volume_unit: base\n    has_ts_recv: false\n",
        encoding="utf-8",
    )
    sources = catalog.load_sources()
    assert "my_csv" in sources
    assert "binance_um_futures" in sources
