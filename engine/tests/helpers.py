"""Synthetic market data and experiments for runner tests.

Each hour has an order-flow imbalance drawn at random; the NEXT hour's return is driven by it
(planted signal) plus noise. A feature reading the current bar's imbalance must find a clearly
positive IC; a feature of pure noise must not.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from trading_engine.core.schema import TRADES_SCHEMA, conform
from trading_engine.data import store

SOURCE, MARKET = "generic_csv", "crypto_perp_usdt"
TRADES_PER_HOUR = 6


def write_assets(workspace: Path, symbols: list[str]) -> None:
    (workspace / "catalog").mkdir(exist_ok=True)
    assets = {s: {"market": MARKET, "source": SOURCE, "streams": ["trades"]} for s in symbols}
    (workspace / "catalog" / "assets.yaml").write_text(yaml.safe_dump(assets), encoding="utf-8")


def write_market(symbols: list[str], start: date, end: date, *, beta: float = 0.004, seed: int = 7) -> None:
    """Trades for [start, end) days, stored as validated daily Parquet with manifest entries."""
    rng = np.random.default_rng(seed)
    hours = int((end - start).days * 24)
    for k, symbol in enumerate(symbols):
        imbalance = rng.uniform(-0.5, 0.5, hours)
        noise = rng.normal(0, 0.002, hours)
        hourly_ret = np.concatenate([[0.0], beta * imbalance[:-1]]) + noise
        log_price = np.log(100.0 * (k + 1)) + np.cumsum(hourly_ret)
        t0 = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        rows = {"trade_id": [], "price": [], "size": [], "aggressor_side": [], "ts_exchange_us": []}
        trade_id = 0
        for h in range(hours):
            hour_start_us = int((t0 + timedelta(hours=h)).timestamp() * 1_000_000)
            buy_share = 0.5 + imbalance[h] / 2
            prev = log_price[h - 1] if h else log_price[0]
            for j in range(TRADES_PER_HOUR):
                frac = (j + 1) / TRADES_PER_HOUR
                rows["trade_id"].append(trade_id)
                rows["price"].append(float(np.exp(prev + frac * (log_price[h] - prev))))
                rows["size"].append(1.0)
                rows["aggressor_side"].append("buy" if j < round(buy_share * TRADES_PER_HOUR) else "sell")
                rows["ts_exchange_us"].append(hour_start_us + j * 500_000_000 + 1)
                trade_id += 1
        df = pl.DataFrame(rows).with_columns((pl.col("price") * pl.col("size")).alias("quote_volume"))
        df = conform(df, TRADES_SCHEMA)
        df = df.with_columns(pl.from_epoch("ts_exchange_us", time_unit="us").dt.date().alias("_day"))
        manifest = store.manifest_path(SOURCE, MARKET, symbol, "trades")
        for (day,), part in df.group_by("_day", maintain_order=True):
            out = store.day_path(SOURCE, MARKET, symbol, "trades", day)
            out.parent.mkdir(parents=True, exist_ok=True)
            part.drop("_day").write_parquet(out)
            store.append_jsonl(manifest, {"day": day.isoformat(), "rows": part.height, "warnings": [],
                                          "raw_sha256": f"{k:02d}{day.isoformat()}"})


FEATURES_OK = '''
import polars as pl


def flow(bars):
    imbalance = (pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("volume")
    return bars.select(imbalance.fill_nan(0.0).alias("imbalance"))


def noise(bars):
    return bars.select((pl.col("n_trades") * 0.0 + (pl.col("ts").dt.hour() % 7)).alias("hour_mod7"))
'''

FEATURES_LOOKAHEAD = '''
import polars as pl


def flow(bars):
    return bars.select((pl.col("close").shift(-1) / pl.col("close") - 1).alias("next_return"))


def noise(bars):
    return bars.select((pl.col("ts").dt.hour() % 7).cast(pl.Float64).alias("hour_mod7"))
'''


def write_experiment(exp_dir: Path, symbols: list[str], *, features: str = FEATURES_OK,
                     holdout_start: str = "2024-03-01", menu: dict | None = None) -> Path:
    exp_dir.mkdir(parents=True)
    family = lambda name, role: {  # noqa: E731
        "name": name, "role": role, "hypothesis": f"{name} hypothesis", "inputs": "bars",
        "lookback": 1, "lag": 0, "failure_mode": f"{name} failure",
    }
    setup = {
        "strategy_id": exp_dir.name, "setup_version": "v1",
        "strategy_map": {"family": "flow_microstructure", "edge_source": "liquidity_provision",
                         "pnl_decomposition": ["price_pnl", "fees"], "failure_modes": ["a", "b"]},
        "universe": {"symbols": symbols, "market": MARKET, "requires": ["trades", "volume"],
                     "eligibility_rule": "always listed"},
        "decision": {"cadence": "1h", "bar_type": "time", "snapshot": "bar_close", "execution_delay": "next_bar_open"},
        "mapping": {"position_state_space": "long_short", "entry_logic": "rank_top_k", "sizing": "equal_weight"},
        "constraints": {"max_gross_leverage": 1.0},
        "costs": {"fee_maker_bps": 2, "fee_taker_bps": 5, "slippage_model": "estimated_half_spread",
                  "sensitivity_bps": [0, 5, 10]},
        "labels": [{"name": "fwd_ret_1h", "horizon": "1h", "target": "return", "primary": True},
                   {"name": "fwd_dir_1h", "horizon": "1h", "target": "direction"}],
        "features": {"families": [family("flow", "signal"), family("noise", "state")]},
        "metrics": {"selection": "rank_ic", "report": ["rank_ic"]},
        "evaluation": {"protocol": "walk_forward", "n_splits": 3, "train_size": "20D", "val_size": "7D",
                       "data_start": "2024-01-01", "holdout": {"start": holdout_start, "end": "2024-04-01"}},
    }
    (exp_dir / "setup.yaml").write_text(yaml.safe_dump(setup, sort_keys=False), encoding="utf-8")
    (exp_dir / "features.py").write_text(features, encoding="utf-8")
    (exp_dir / "training").mkdir()
    menus = menu or {
        "fwd_ret_1h": {"linear": {"ridge_a1": {"estimator": "ridge", "alpha": 1.0}},
                       "gbm": {"lgbm_small": {"estimator": "lightgbm", "num_leaves": 7, "n_estimators": 50}}},
        "fwd_dir_1h": {"linear": {"logit": {"estimator": "logistic", "C": 1.0}}},
    }
    for label, models in menus.items():
        (exp_dir / "training" / f"{label}.yaml").write_text(
            yaml.safe_dump({"label": label, "models": models}), encoding="utf-8")
    sections = "\n\n".join(f"## {h}\nFilled." for h in
                           ("Hypothesis", "Economic mechanism", "Pre-registration", "Success criteria", "Kill criteria"))
    (exp_dir / "research.md").write_text(f"# {exp_dir.name}\n\n{sections}\n", encoding="utf-8")
    return exp_dir


def runlog_rows(exp_dir: Path) -> list[dict]:
    import sqlite3
    db = sqlite3.connect(exp_dir / "run_log" / "registry.db")
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY label, config_name")]
    finally:
        db.close()
