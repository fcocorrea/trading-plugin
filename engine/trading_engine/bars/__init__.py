"""Bars built causally from normalized trades. A bar is stamped with its close time: the bar at t
covers trades in [t - cadence, t) and is known at t, which is the decision time.

Only time bars exist for now. Tick, volume and dollar bars (ML4T ch03) raise UnimplementedError.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from ..core.errors import DataValidationError, UnimplementedError

BAR_COLUMNS = (
    "symbol", "ts", "open", "high", "low", "close", "vwap",
    "volume", "quote_volume", "buy_volume", "sell_volume", "n_trades",
)


def _every(cadence: timedelta) -> str:
    return f"{int(cadence.total_seconds() // 60)}m"


def time_bars(trades: pl.DataFrame, symbol: str, cadence: timedelta) -> pl.DataFrame:
    """Time bars on a complete grid. A bar without trades keeps the previous close as its
    open/high/low/close and has zero volume, so every decision time exists."""
    if trades.is_empty():
        raise DataValidationError(f"{symbol}: no trades to build bars")
    every = _every(cadence)
    side = pl.col("aggressor_side")
    bars = (
        trades.lazy()
        .with_columns(pl.from_epoch("ts_exchange_us", time_unit="us").dt.replace_time_zone("UTC").alias("t"))
        .sort("t")
        .group_by_dynamic("t", every=every, closed="left", label="right")
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("size").filter(side == "buy").sum().alias("buy_volume"),
            pl.col("size").filter(side == "sell").sum().alias("sell_volume"),
            pl.len().cast(pl.Int64).alias("n_trades"),
        )
        .rename({"t": "ts"})
        .collect()
    )
    grid = pl.DataFrame({
        "ts": pl.datetime_range(bars["ts"].min(), bars["ts"].max(), interval=every, eager=True, time_zone="UTC"),
    })
    filled = (
        grid.join(bars, on="ts", how="left")
        .with_columns(pl.col("close").forward_fill())
        .with_columns(
            [pl.col(c).fill_null(pl.col("close")) for c in ("open", "high", "low")]
            + [pl.col(c).fill_null(0) for c in ("volume", "quote_volume", "buy_volume", "sell_volume", "n_trades")]
        )
        .with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("quote_volume") / pl.col("volume"))
            .otherwise(pl.col("close"))
            .alias("vwap"),
            pl.lit(symbol).alias("symbol"),
        )
    )
    return filled.select(BAR_COLUMNS)


def build_bars(trades: pl.DataFrame, symbol: str, bar_type: str, cadence: timedelta) -> pl.DataFrame:
    if bar_type != "time":
        raise UnimplementedError(f"bar_type '{bar_type}' is not implemented yet; only time bars exist")
    return time_bars(trades, symbol, cadence)
