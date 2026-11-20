"""Time: unit detection, normalization to microseconds, day ranges and durations."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import polars as pl

from .errors import DataValidationError, TradingError

# Magnitude thresholds, largest first. A current timestamp in ms is about 1.7e12.
_THRESHOLDS = ((10**17, "ns"), (10**14, "us"), (10**11, "ms"), (10**8, "s"))
_TO_US = {"us": 1, "ms": 1_000, "s": 1_000_000}
_US_PER_DAY = 86_400 * 1_000_000

_DURATION_RE = re.compile(r"^\s*(\d+)\s*(min|h|d|w|y)\s*$", re.IGNORECASE)
_DURATION_UNITS = {
    "min": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
    "y": timedelta(days=365),
}


def detect_unit(sample_max: int) -> str:
    """Detect the unit by magnitude. Call once per file, with the column maximum."""
    for threshold, unit in _THRESHOLDS:
        if sample_max >= threshold:
            return unit
    raise DataValidationError(f"timestamp outside any plausible range: {sample_max}")


def to_us(col: pl.Expr, unit: str) -> pl.Expr:
    if unit == "ns":
        return (col // 1000).cast(pl.Int64)
    return (col * _TO_US[unit]).cast(pl.Int64)


def days_between(start: date, end: date) -> list[date]:
    """Inclusive list of days between start and end."""
    if end < start:
        raise TradingError(f"end ({end}) is before start ({start})")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def day_bounds_us(day: date) -> tuple[int, int]:
    """Half-open [start, end) interval of the UTC day, in microseconds."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    start_us = int(start.timestamp()) * 1_000_000
    return start_us, start_us + _US_PER_DAY


def parse_duration(value: object) -> timedelta:
    """Parse durations such as '8h', '30D', '2Y' or '15min'. A year is 365 days (24/7 calendar)."""
    match = _DURATION_RE.match(str(value))
    if not match:
        raise TradingError(f"invalid duration '{value}': use <n><min|h|D|W|Y>, e.g. 8h, 90D, 1Y")
    amount, unit = int(match.group(1)), match.group(2).lower()
    if amount <= 0:
        raise TradingError(f"duration '{value}' must be positive")
    return amount * _DURATION_UNITS[unit]
