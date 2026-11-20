"""Normalized schema. Every source produces exactly these columns and types."""
from __future__ import annotations

import polars as pl

from .errors import UnimplementedError

TRADES_SCHEMA: dict = {
    "trade_id": pl.Int64,         # null when the source has no identifiers
    "price": pl.Float64,
    "size": pl.Float64,           # in the unit declared by Capabilities.volume_unit
    "quote_volume": pl.Float64,   # notional in quote currency (see quote_volume_expr)
    "aggressor_side": pl.Utf8,    # "buy" | "sell" | null
    "ts_exchange_us": pl.Int64,
    "ts_recv_us": pl.Int64,       # null when the source does not provide it
}

QUOTES_SCHEMA: dict = {
    "ts_exchange_us": pl.Int64,
    "ts_recv_us": pl.Int64,
    "bid_px": pl.Float64,
    "bid_sz": pl.Float64,
    "ask_px": pl.Float64,
    "ask_sz": pl.Float64,
}

AGGRESSOR_SIDES = ("buy", "sell")


def quote_volume_expr(volume_unit: str) -> pl.Expr:
    """Notional in quote currency from `size`, according to the unit it is expressed in."""
    if volume_unit == "base":
        return pl.col("price") * pl.col("size")
    if volume_unit == "quote":
        return pl.col("size")
    if volume_unit == "contracts":
        raise UnimplementedError(
            "volume_unit 'contracts' needs the contract multiplier to compute the notional"
        )
    raise UnimplementedError(f"volume_unit '{volume_unit}' does not produce traded volume")


def conform(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """Add missing columns as null, cast types and order the columns."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=schema[c]).alias(c) for c in missing])
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
