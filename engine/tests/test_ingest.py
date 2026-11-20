from datetime import date

import pytest

from trading_engine.core.timeutil import day_bounds_us
from trading_engine.data import store
from trading_engine.data.ingest import ingest

SOURCE, MARKET, SYMBOL = "generic_csv", "crypto_perp_usdt", "TESTUSDT"
D1, D2, D3 = date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)


@pytest.fixture
def workspace(isolated_env):
    (isolated_env / "catalog").mkdir()
    (isolated_env / "catalog" / "assets.yaml").write_text(
        f"{SYMBOL}:\n  market: {MARKET}\n  source: {SOURCE}\n  streams: [trades]\n", encoding="utf-8"
    )
    return isolated_env


def _write_day(day: date, body_rows: list[str]) -> None:
    raw = store.raw_dir(SOURCE, SYMBOL, "trades")
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{day.isoformat()}.csv").write_text("ts,price,size\n" + "\n".join(body_rows) + "\n")


def _ms(day: date, offset: int = 0) -> int:
    return day_bounds_us(day)[0] // 1000 + offset


def test_bad_day_does_not_abort_the_range(workspace):
    _write_day(D1, [f"{_ms(D1)},100,1"])
    _write_day(D2, ["not-a-timestamp,100,1"])
    _write_day(D3, [f"{_ms(D3)},100,1"])

    summary = ingest(SYMBOL, D1, D3)

    assert [d["day"] for d in summary["ok"]] == ["2025-01-01", "2025-01-03"]
    assert [d["day"] for d in summary["failed"]] == ["2025-01-02"]
    assert not store.day_path(SOURCE, MARKET, SYMBOL, "trades", D2).exists()


def test_file_with_rows_from_another_day_fails(workspace):
    _write_day(D1, [f"{_ms(D2)},100,1"])
    summary = ingest(SYMBOL, D1, D1)
    assert summary["failed"] and "outside the UTC day" in summary["failed"][0]["error"]


def test_parquet_without_manifest_entry_is_reingested(workspace):
    _write_day(D1, [f"{_ms(D1)},100,1"])
    out = store.day_path(SOURCE, MARKET, SYMBOL, "trades", D1)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"half-written parquet")

    summary = ingest(SYMBOL, D1, D1)

    assert [d["day"] for d in summary["ok"]] == ["2025-01-01"]
    assert store.manifest_days(store.manifest_path(SOURCE, MARKET, SYMBOL, "trades")) == {"2025-01-01"}


def test_second_run_uses_cache_and_leaves_no_part_files(workspace):
    _write_day(D1, [f"{_ms(D1)},100,1"])
    ingest(SYMBOL, D1, D1)
    summary = ingest(SYMBOL, D1, D1)
    assert summary["cached"] == ["2025-01-01"]
    assert not list(store.data_dir(SOURCE, MARKET, SYMBOL, "trades").glob("*.part"))
