"""Your own CSV files in data/raw/generic_csv/{SYMBOL}/{stream}/{YYYY-MM-DD}.csv (any market)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from ..core.errors import DataValidationError
from ..core.interfaces import Capabilities
from ..core.schema import QUOTES_SCHEMA, TRADES_SCHEMA, conform, quote_volume_expr
from ..core.timeutil import days_between, detect_unit, to_us


def _read_csv(path: Path, required: list[str]) -> pl.DataFrame:
    if not path.exists():
        raise DataValidationError(f"missing file {path}. Put it at that path (see the add-source skill)")
    df = pl.read_csv(path, infer_schema_length=10_000)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {missing}")
    if df.height == 0:
        raise DataValidationError(f"{path.name}: file without rows")
    return df


class GenericCSV:
    def __init__(self, config: dict):
        self.name = "generic_csv"
        self.capabilities = Capabilities(**config["capabilities"])

    def list_days(self, symbol: str, start: date, end: date) -> list[date]:
        return days_between(start, end)

    def fetch_raw(self, symbol: str, stream: str, day: date, raw_dir: Path) -> Path:
        # The CSV files are already on disk: "fetch" only locates the file.
        return raw_dir / f"{day.isoformat()}.csv"

    def to_trades(self, raw: Path) -> pl.DataFrame:
        df = _read_csv(raw, ["ts", "price", "size"])
        has_id = "trade_id" in df.columns
        has_side = "side" in df.columns
        unit = detect_unit(int(df["ts"].cast(pl.Int64).max()))
        out = df.select(
            (pl.col("trade_id") if has_id else pl.lit(None)).cast(pl.Int64).alias("trade_id"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            to_us(pl.col("ts").cast(pl.Int64), unit).alias("ts_exchange_us"),
            (pl.col("side").cast(pl.Utf8).str.to_lowercase() if has_side
             else pl.lit(None, dtype=pl.Utf8)).alias("aggressor_side"),
        ).with_columns(quote_volume_expr(self.capabilities.volume_unit).alias("quote_volume"))
        return conform(out, TRADES_SCHEMA)

    def to_quotes(self, raw: Path) -> pl.DataFrame:
        df = _read_csv(raw, ["ts", "bid_px", "bid_sz", "ask_px", "ask_sz"])
        unit = detect_unit(int(df["ts"].cast(pl.Int64).max()))
        out = df.select(
            to_us(pl.col("ts").cast(pl.Int64), unit).alias("ts_exchange_us"),
            pl.col("bid_px").cast(pl.Float64),
            pl.col("bid_sz").cast(pl.Float64),
            pl.col("ask_px").cast(pl.Float64),
            pl.col("ask_sz").cast(pl.Float64),
        )
        return conform(out, QUOTES_SCHEMA)
