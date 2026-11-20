"""Binance USD-M futures: historical aggTrades from data.binance.vision (public, free)."""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import requests

from ..core.errors import DataValidationError, UnimplementedError
from ..core.interfaces import Capabilities
from ..core.schema import TRADES_SCHEMA, conform, quote_volume_expr
from ..core.timeutil import days_between, detect_unit, to_us

RAW_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


class BinanceUMFutures:
    def __init__(self, config: dict):
        self.name = "binance_um_futures"
        self.base_url = config["base_url"].rstrip("/")
        self.capabilities = Capabilities(**config["capabilities"])
        self._datasets = config.get("datasets", {})

    def list_days(self, symbol: str, start: date, end: date) -> list[date]:
        return days_between(start, end)

    def _dataset(self, stream: str) -> str:
        if stream != "trades":
            raise UnimplementedError(
                "historical bookTicker is not verified for this adapter. "
                "Use `trading-engine record` to record live quotes."
            )
        return self._datasets["trades"]

    def _url(self, dataset: str, symbol: str, day: date, suffix: str = "") -> str:
        return (f"{self.base_url}/daily/{dataset}/{symbol}/"
                f"{symbol}-{dataset}-{day.isoformat()}.zip{suffix}")

    def fetch_raw(self, symbol: str, stream: str, day: date, raw_dir: Path) -> Path:
        dataset = self._dataset(stream)
        raw_dir.mkdir(parents=True, exist_ok=True)
        zpath = raw_dir / f"{symbol}-{dataset}-{day.isoformat()}.zip"
        if zpath.exists():
            return zpath
        r = requests.get(self._url(dataset, symbol, day), timeout=120)
        r.raise_for_status()
        c = requests.get(self._url(dataset, symbol, day, ".CHECKSUM"), timeout=30)
        c.raise_for_status()
        expected = c.text.split()[0].lower()
        if hashlib.sha256(r.content).hexdigest() != expected:
            raise DataValidationError(f"checksum mismatch for {symbol} {day}")
        tmp = zpath.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(zpath)
        return zpath

    def _read_raw(self, zpath: Path) -> pl.DataFrame:
        with zipfile.ZipFile(zpath) as z:
            raw = z.read(z.namelist()[0])
        has_header = raw[:20].lower().startswith(b"agg_trade_id")
        # infer_schema_length=0: everything as text; casting is explicit, not guessed
        df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema_length=0)
        if df.width != len(RAW_COLS):
            raise DataValidationError(
                f"{zpath.name}: {df.width} columns, expected {len(RAW_COLS)}"
            )
        df.columns = RAW_COLS
        return df

    def to_trades(self, raw: Path) -> pl.DataFrame:
        df = self._read_raw(raw)
        if df.height == 0:
            raise DataValidationError(f"{raw.name}: file without trades")
        unit = detect_unit(int(df["transact_time"].cast(pl.Int64).max()))
        out = df.select(
            pl.col("agg_trade_id").cast(pl.Int64).alias("trade_id"),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64).alias("size"),
            to_us(pl.col("transact_time").cast(pl.Int64), unit).alias("ts_exchange_us"),
            # is_buyer_maker=true: the buyer was the maker, so the aggressor sold
            pl.when(pl.col("is_buyer_maker").str.to_lowercase() == "true")
              .then(pl.lit("sell"))
              .otherwise(pl.lit("buy"))
              .alias("aggressor_side"),
        ).with_columns(quote_volume_expr(self.capabilities.volume_unit).alias("quote_volume"))
        return conform(out, TRADES_SCHEMA)

    def to_quotes(self, raw: Path) -> pl.DataFrame:
        raise UnimplementedError("historical bookTicker is not available in this adapter")
