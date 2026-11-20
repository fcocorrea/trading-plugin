"""Model and signal diagnostics on out-of-sample predictions (ML4T ch06, ch07, ch11).

Signal quality is the rank IC with Newey-West (HAC) inference from ml4t-diagnostic, the same
functions the book's case studies use (cross_sectional_ic_series, compute_ic_hac_stats).

IC mode follows the book's rule (case-study configs: incremental_ic.min_cross_section = 10):
- cross_sectional: at least MIN_CROSS_SECTION symbols per decision time -> one IC per timestamp,
  ranking symbols against each other.
- time_series: smaller universes -> one IC per symbol and period (the shortest of 1D, 7D, 28D that
  holds at least MIN_PERIOD_OBS bars), averaged across symbols per period. It measures whether
  the score ranks a symbol's own future returns over time.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import polars as pl

MIN_CROSS_SECTION = 10
MIN_PERIOD_OBS = 10
TS_PERIODS = (timedelta(days=1), timedelta(days=7), timedelta(days=28))


@dataclass(frozen=True)
class SignalStats:
    mode: str
    period: str
    mean_ic: float
    t_stat: float
    p_value: float
    n_periods: int
    hac_lags: int
    ic_ir: float
    positive_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def ic_mode(frame: pl.DataFrame) -> str:
    per_ts = frame.group_by("ts").agg(pl.col("symbol").n_unique().alias("n"))
    return "cross_sectional" if per_ts["n"].median() >= MIN_CROSS_SECTION else "time_series"


def _period_for(cadence: timedelta) -> timedelta:
    for period in TS_PERIODS:
        if period // cadence >= MIN_PERIOD_OBS:
            return period
    return TS_PERIODS[-1]


def ic_series(frame: pl.DataFrame, label: str, cadence: timedelta, mode: str) -> tuple[pl.DataFrame, timedelta]:
    """IC per period with columns [period, ic]. frame has ts, symbol, prediction and the label."""
    from ml4t.diagnostic.metrics import cross_sectional_ic_series

    data = frame.select("ts", "symbol", "prediction", pl.col(label).alias("forward_return")).drop_nulls()
    if mode == "cross_sectional":
        series = cross_sectional_ic_series(
            data, data, pred_col="prediction", ret_col="forward_return", date_col="ts",
            entity_col="symbol", min_obs=MIN_CROSS_SECTION,
        )
        return series.select(pl.col("ts").alias("period"), "ic").drop_nulls(), cadence

    period = _period_for(cadence)
    every = f"{int(period.total_seconds() // 60)}m"
    keyed = data.with_columns(
        pl.col("ts").dt.truncate(every).alias("period"),
        pl.int_range(pl.len()).alias("row"),
    ).with_columns((pl.col("period").cast(pl.Utf8) + "|" + pl.col("symbol")).alias("bucket"))
    per_symbol = cross_sectional_ic_series(
        keyed, keyed, pred_col="prediction", ret_col="forward_return", date_col="bucket",
        entity_col="row", min_obs=MIN_PERIOD_OBS,
    )
    series = (
        per_symbol.drop_nulls("ic")
        .with_columns(pl.col("bucket").str.split("|").list.first().alias("period"))
        .group_by("period").agg(pl.col("ic").mean())
        .sort("period")
    )
    return series.select("period", "ic"), period


def signal_stats(series: pl.DataFrame, mode: str, period: timedelta, horizon: timedelta) -> SignalStats:
    from ml4t.diagnostic.metrics import compute_ic_hac_stats

    ics = series["ic"].to_numpy()
    label_periods = max(1, math.ceil(horizon / period))
    if len(ics) < 3:
        return SignalStats(mode, _fmt(period), float(np.mean(ics)) if len(ics) else math.nan,
                           math.nan, math.nan, len(ics), 0, math.nan, math.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hac = compute_ic_hac_stats(series, ic_col="ic", label_horizon=label_periods)
    std = float(np.std(ics, ddof=1))
    return SignalStats(
        mode=mode,
        period=_fmt(period),
        # ml4t-diagnostic 0.2.1 returns a plain dict here, despite its ICHACStats annotation.
        mean_ic=float(hac["mean_ic"]),
        t_stat=float(hac["t_stat"]),
        p_value=float(hac["p_value"]),
        n_periods=int(hac["n_periods"]),
        hac_lags=int(hac["effective_lags"]),
        ic_ir=float(np.mean(ics) / std) if std > 0 else math.nan,
        positive_share=float(np.mean(ics > 0)),
    )


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    from sklearn import metrics

    if target == "direction":
        out = {"log_loss": float(metrics.log_loss(y_true, np.clip(y_pred, 1e-12, 1 - 1e-12), labels=[0, 1]))}
        out["auc"] = float(metrics.roc_auc_score(y_true, y_pred)) if len(set(y_true)) == 2 else math.nan
        out["hit_rate"] = float(np.mean((y_pred > 0.5) == (y_true == 1)))
        return out
    return {
        "mse": float(metrics.mean_squared_error(y_true, y_pred)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "hit_rate": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def _fmt(period: timedelta) -> str:
    minutes = int(period.total_seconds() // 60)
    if minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"
