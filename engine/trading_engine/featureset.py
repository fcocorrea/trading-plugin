"""Feature families computed by the experiment's own features.py (ML4T ch08).

Contract: features.py defines one function per family declared in setup.yaml, with the family's
name. It receives the bars of ONE symbol, sorted by ts, with the columns of bars.BAR_COLUMNS, and
returns a DataFrame with the same number of rows and one column per feature. The runner prefixes
the columns with "<family>__", applies the family's lag, and drops warm-up rows with nulls.

Every family is checked for look-ahead before it is used: it is recomputed on truncated histories
and the values before each cut must not change. A feature that changes when the future is removed
was reading the future.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl

from .core.errors import TradingError

FEATURES_FILE = "features.py"
SEPARATOR = "__"
CAUSALITY_CUTS = (0.35, 0.6, 0.85)
TOLERANCE = 1e-9


class LookAheadError(TradingError):
    """A feature family uses data that was not available at its timestamp."""


def module_path(exp_dir: Path) -> Path:
    return exp_dir / FEATURES_FILE


def module_digest(exp_dir: Path) -> str:
    return hashlib.sha256(module_path(exp_dir).read_bytes()).hexdigest()


def load_module(exp_dir: Path) -> ModuleType:
    path = module_path(exp_dir)
    if not path.is_file():
        raise TradingError(f"missing {path}: define one function per feature family declared in setup.yaml")
    spec = importlib.util.spec_from_file_location(f"experiment_features_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 - user code: report any failure as a validation error
        raise TradingError(f"{FEATURES_FILE} failed to import: {type(e).__name__}: {e}") from e
    return module


def check_families(module: ModuleType, families: list[dict]) -> None:
    missing = [f["name"] for f in families if not callable(getattr(module, f["name"], None))]
    if missing:
        raise TradingError(f"{FEATURES_FILE} has no function for the declared families {missing}")


def _compute(module: ModuleType, family: dict, bars: pl.DataFrame) -> pl.DataFrame:
    name = family["name"]
    try:
        out = getattr(module, name)(bars)
    except Exception as e:  # noqa: BLE001 - user code
        raise TradingError(f"feature family {name} failed: {type(e).__name__}: {e}") from e
    if not isinstance(out, pl.DataFrame) or out.width == 0:
        raise TradingError(f"feature family {name} must return a non-empty polars DataFrame")
    if out.height != bars.height:
        raise TradingError(f"feature family {name} returned {out.height} rows for {bars.height} bars")
    return out.select([pl.col(c).cast(pl.Float64).alias(f"{name}{SEPARATOR}{c}") for c in out.columns])


def _same(a: pl.Series, b: pl.Series) -> bool:
    a, b = a.fill_nan(None), b.fill_nan(None)
    both_null = a.is_null() & b.is_null()
    close = (a - b).abs() <= TOLERANCE
    return bool((both_null | close.fill_null(False)).all())


def check_causality(module: ModuleType, family: dict, bars: pl.DataFrame) -> None:
    full = _compute(module, family, bars)
    for fraction in CAUSALITY_CUTS:
        cut = int(bars.height * fraction)
        if cut <= family["lookback"] + 1:
            continue
        truncated = _compute(module, family, bars.head(cut))
        for column in full.columns:
            if not _same(full[column].head(cut), truncated[column]):
                raise LookAheadError(
                    f"feature {column} changes when data after {bars['ts'][cut - 1]} is removed: "
                    "it reads the future"
                )


def compute_features(module: ModuleType, families: list[dict], bars: pl.DataFrame) -> pl.DataFrame:
    """Feature columns for one symbol's bars, lagged per family, aligned row by row with the bars."""
    blocks = []
    for family in families:
        block = _compute(module, family, bars)
        if family["lag"]:
            block = block.select(pl.all().shift(family["lag"]))
        blocks.append(block)
    return pl.concat(blocks, how="horizontal")
