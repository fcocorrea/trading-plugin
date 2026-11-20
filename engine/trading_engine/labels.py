"""Forward labels consistent with execution (ML4T ch07).

With execution_delay = next_bar_open, a decision at bar t (stamped with its close) enters at the
open of the next bar and exits h later, at the open of the bar that starts at t + h:

    fwd_ret_h(t) = open[t + cadence + h] / open[t + cadence] - 1

which covers the price path over [t, t + h]. The label is known at t + h, so a training row at t is
admissible for a validation window starting at v only if t + h < v: that is the purge.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

from .core.errors import UnimplementedError
from .experiment import Label


def add_labels(bars: pl.DataFrame, labels: list[Label], cadence: timedelta, execution_delay: str) -> pl.DataFrame:
    """Add one column per label to single-symbol bars sorted by ts. Rows whose exit lies beyond the
    loaded data get a null label."""
    if execution_delay != "next_bar_open":
        raise UnimplementedError(f"execution_delay '{execution_delay}' is not implemented yet; use next_bar_open")
    entry = pl.col("open").shift(-1)
    columns = []
    for label in labels:
        steps = label.horizon // cadence
        exit_ = pl.col("open").shift(-(1 + steps))
        ret = exit_ / entry - 1
        if label.target == "return":
            columns.append(ret.alias(label.name))
        else:
            columns.append(pl.when(ret.is_null()).then(None).otherwise((ret > 0).cast(pl.Int8)).alias(label.name))
    return bars.with_columns(columns)
