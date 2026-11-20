from datetime import date

import polars as pl
import pytest

from trading_engine.core.errors import DataValidationError, TradingError, UnimplementedError
from trading_engine.core.interfaces import Capabilities
from trading_engine.core.schema import TRADES_SCHEMA, conform, quote_volume_expr
from trading_engine.core.timeutil import day_bounds_us, days_between, detect_unit
from trading_engine.data.validate import validate_quotes, validate_trades

DAY = date(2025, 1, 1)
DAY_START_US, _ = day_bounds_us(DAY)
CAPS = Capabilities(trades=True, quotes=False, aggressor_side=True, volume_unit="base", has_ts_recv=False)
NO_SIDE_CAPS = Capabilities(trades=True, quotes=False, aggressor_side=False, volume_unit="base", has_ts_recv=False)


def _trades(sizes, ts=None, sides=None, ids=None):
    n = len(sizes)
    df = pl.DataFrame({
        "trade_id": ids if ids is not None else list(range(n)),
        "price": [100.0] * n,
        "size": sizes,
        "aggressor_side": sides if sides is not None else ["buy"] * n,
        "ts_exchange_us": ts if ts is not None else [DAY_START_US + 1_000_000 * i for i in range(n)],
    })
    return conform(df, TRADES_SCHEMA)


def test_detect_unit():
    assert detect_unit(1_700_000_000) == "s"
    assert detect_unit(1_700_000_000_000) == "ms"
    assert detect_unit(1_700_000_000_000_000) == "us"
    assert detect_unit(1_700_000_000_000_000_000) == "ns"


def test_detect_unit_rejects_implausible_value():
    with pytest.raises(DataValidationError):
        detect_unit(42)


def test_days_between_rejects_inverted_range():
    with pytest.raises(TradingError):
        days_between(date(2025, 1, 2), date(2025, 1, 1))


def test_day_bounds_are_utc_midnight():
    start, end = day_bounds_us(DAY)
    assert start == 1_735_689_600_000_000
    assert end - start == 86_400_000_000


def test_quote_volume_by_unit():
    df = pl.DataFrame({"price": [100.0], "size": [2.0]})
    assert df.select(quote_volume_expr("base")).item() == 200.0
    assert df.select(quote_volume_expr("quote")).item() == 2.0
    with pytest.raises(UnimplementedError):
        quote_volume_expr("contracts")


def test_valid_trades_pass():
    assert validate_trades(_trades([1.0, 2.0, 0.5]), CAPS, DAY).ok


def test_zero_size_rejected():
    assert not validate_trades(_trades([1.0, 0.0]), CAPS, DAY).ok


def test_no_volume_source_rejected():
    no_vol = Capabilities(trades=True, quotes=False, aggressor_side=False, volume_unit="none", has_ts_recv=False)
    assert not validate_trades(_trades([1.0]), no_vol, DAY).ok


def test_backwards_time_rejected():
    ts = [DAY_START_US + 2_000_000, DAY_START_US + 1_000_000]
    assert not validate_trades(_trades([1.0, 1.0], ts=ts), CAPS, DAY).ok


def test_rows_outside_file_day_rejected():
    previous_day = DAY_START_US - 1
    rep = validate_trades(_trades([1.0], ts=[previous_day]), CAPS, DAY)
    assert not rep.ok
    assert "outside the UTC day" in rep.errors[0]


def test_missing_side_rejected_when_source_declares_it():
    assert not validate_trades(_trades([1.0], sides=[None]), CAPS, DAY).ok
    assert validate_trades(_trades([1.0], sides=[None]), NO_SIDE_CAPS, DAY).ok


def test_unknown_side_value_rejected():
    assert not validate_trades(_trades([1.0], sides=["b"]), NO_SIDE_CAPS, DAY).ok


def test_null_trade_ids_skip_id_checks_with_warning():
    rep = validate_trades(_trades([1.0, 1.0], ids=[None, None]), CAPS, DAY, prev_last_id=10)
    assert rep.ok
    assert rep.warnings == ["no trade_id: duplicates and gaps cannot be detected"]


def test_partial_null_trade_ids_rejected():
    assert not validate_trades(_trades([1.0, 1.0], ids=[1, None]), CAPS, DAY).ok


def test_duplicate_trade_ids_rejected():
    assert not validate_trades(_trades([1.0, 1.0], ids=[5, 5]), CAPS, DAY).ok


def test_day_boundary_gap_warns():
    rep = validate_trades(_trades([1.0], ids=[7]), CAPS, DAY, prev_last_id=5)
    assert rep.ok
    assert "gap at the boundary with the previous day" in rep.warnings


def test_quotes_outside_day_rejected():
    caps = Capabilities(trades=True, quotes=True, aggressor_side=False, volume_unit="base", has_ts_recv=False)
    q = pl.DataFrame({
        "ts_exchange_us": [DAY_START_US - 1],
        "bid_px": [99.0], "bid_sz": [1.0], "ask_px": [101.0], "ask_sz": [1.0],
    })
    assert not validate_quotes(q, caps, DAY).ok
