"""Engine contracts. Every new source or market implements these interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

VOLUME_UNITS = ("base", "quote", "contracts")


@dataclass(frozen=True)
class Capabilities:
    """What a source actually delivers. Compared against the requirements of the experiment."""

    trades: bool
    quotes: bool
    aggressor_side: bool
    volume_unit: str  # "base" | "quote" | "contracts" | "none"
    has_ts_recv: bool

    @property
    def has_volume(self) -> bool:
        return self.trades and self.volume_unit in VOLUME_UNITS


@runtime_checkable
class Source(Protocol):
    name: str
    capabilities: Capabilities

    def list_days(self, symbol: str, start: date, end: date) -> list[date]: ...

    def fetch_raw(self, symbol: str, stream: str, day: date, raw_dir: Path) -> Path: ...

    def to_trades(self, raw: Path) -> pl.DataFrame: ...

    def to_quotes(self, raw: Path) -> pl.DataFrame: ...


class MarketRules(Protocol):
    def is_tradable(self, ts_us: int) -> bool: ...

    def can_short(self, symbol: str, ts_us: int) -> bool: ...

    def fee(self, notional: float, side: str, liquidity: str) -> float: ...

    def settlement_events(self, start_us: int, end_us: int) -> list[dict]: ...
