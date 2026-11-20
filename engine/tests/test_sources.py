import zipfile
from datetime import date

import pytest

from trading_engine.catalog import load_sources
from trading_engine.core.errors import DataValidationError
from trading_engine.core.timeutil import day_bounds_us
from trading_engine.sources.binance_um import BinanceUMFutures
from trading_engine.sources.generic_csv import GenericCSV

DAY = date(2025, 1, 1)
DAY_START_MS = day_bounds_us(DAY)[0] // 1000
HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"


def _binance_zip(tmp_path, body: str):
    zpath = tmp_path / "XYZUSDT-aggTrades-2025-01-01.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("XYZUSDT-aggTrades-2025-01-01.csv", body)
    return zpath


@pytest.fixture
def binance():
    return BinanceUMFutures(load_sources()["binance_um_futures"])


def test_binance_aggressor_side_from_is_buyer_maker(tmp_path, binance):
    body = HEADER + (
        f"1,100.5,2.0,10,11,{DAY_START_MS},true\n"
        f"2,100.6,0.5,12,12,{DAY_START_MS + 1},false\n"
    )
    df = binance.to_trades(_binance_zip(tmp_path, body))
    # is_buyer_maker=true: the buyer was the maker, so the aggressor was the seller
    assert df["aggressor_side"].to_list() == ["sell", "buy"]
    assert df["trade_id"].to_list() == [1, 2]
    assert df["ts_exchange_us"].to_list() == [DAY_START_MS * 1000, (DAY_START_MS + 1) * 1000]
    assert df["quote_volume"].to_list() == [201.0, 50.3]


def test_binance_file_without_header(tmp_path, binance):
    body = f"1,100.0,1.0,1,1,{DAY_START_MS},True\n"
    df = binance.to_trades(_binance_zip(tmp_path, body))
    assert df["aggressor_side"].to_list() == ["sell"]


def test_binance_microsecond_timestamps(tmp_path, binance):
    body = HEADER + f"1,100.0,1.0,1,1,{DAY_START_MS * 1000 + 7},false\n"
    df = binance.to_trades(_binance_zip(tmp_path, body))
    assert df["ts_exchange_us"].to_list() == [DAY_START_MS * 1000 + 7]


def test_binance_unexpected_column_count(tmp_path, binance):
    with pytest.raises(DataValidationError):
        binance.to_trades(_binance_zip(tmp_path, "1,2,3\n"))


def _csv_source(**caps):
    base = dict(trades=True, quotes=True, aggressor_side=False, volume_unit="base", has_ts_recv=False)
    return GenericCSV({"capabilities": {**base, **caps}})


def test_generic_csv_quote_unit_volume(tmp_path):
    p = tmp_path / "2025-01-01.csv"
    p.write_text(f"ts,price,size\n{DAY_START_MS},100000,5000\n")
    df = _csv_source(volume_unit="quote").to_trades(p)
    assert df["quote_volume"].to_list() == [5000.0]


def test_generic_csv_without_ids_keeps_them_null(tmp_path):
    p = tmp_path / "2025-01-01.csv"
    p.write_text(f"ts,price,size\n{DAY_START_MS},100,1\n{DAY_START_MS + 1},100,1\n")
    df = _csv_source().to_trades(p)
    assert df["trade_id"].null_count() == 2


def test_generic_csv_side_is_normalized(tmp_path):
    p = tmp_path / "2025-01-01.csv"
    p.write_text(f"ts,price,size,side\n{DAY_START_MS},100,1,BUY\n")
    assert _csv_source().to_trades(p)["aggressor_side"].to_list() == ["buy"]


def test_generic_csv_missing_file(tmp_path):
    with pytest.raises(DataValidationError):
        _csv_source().to_trades(tmp_path / "2025-01-01.csv")
