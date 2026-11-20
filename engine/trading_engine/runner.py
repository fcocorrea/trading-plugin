"""Research runner, phase 1: data -> bars -> features -> labels -> walk-forward models -> model and
signal diagnostics -> run log (ML4T ch06-ch12).

Strategy simulation, costs, the cost sweep and the Deflated Sharpe are phase 2. The sealed holdout
is never read: data is loaded strictly before its start.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import __version__, catalog, diagnostics, featureset, models
from .bars import build_bars
from .core.errors import TradingError
from .data.load import load_trades
from .experiment import Experiment, Fold, Label, load_experiment
from .labels import add_labels
from .runlog import RunLog, run_id

MIN_TRAIN_ROWS = 50
# Sections of setup.yaml that change what a model run computes. Costs, mapping, constraints and
# metrics do not affect training, so editing them does not invalidate trained runs.
IDENTITY_SECTIONS = ("setup_version", "universe", "decision", "labels", "features", "evaluation")


@dataclass
class TrialResult:
    label: str
    model_family: str
    config_name: str
    run_id: str
    status: str
    reused: bool = False
    error: str | None = None
    signal: dict | None = None
    model: dict | None = None
    n_oos: int = 0


@dataclass
class RunReport:
    experiment: Experiment
    trials: list[TrialResult] = field(default_factory=list)
    dropped_rows: dict[str, int] = field(default_factory=dict)


def _build_panel(exp: Experiment, module, progress: Callable[[str], None]) -> tuple[pl.DataFrame, list[str], dict]:
    setup = exp.setup
    universe, decision = setup["universe"], setup["decision"]
    families = setup["features"]["families"]
    start, end = exp.data_start, exp.holdout[0]
    parts, fingerprint = [], {}
    for symbol in universe["symbols"]:
        res = catalog.resolve(str(symbol), requires=universe["requires"])
        trades, days = load_trades(res, start, end)
        bars = build_bars(trades, res.symbol, decision["bar_type"], exp.cadence)
        for family in families:
            featureset.check_causality(module, family, bars)
        feats = featureset.compute_features(module, families, bars)
        labeled = add_labels(bars, exp.labels, exp.cadence, decision["execution_delay"])
        parts.append(pl.concat(
            [labeled.select("ts", "symbol", *[label.name for label in exp.labels]), feats], how="horizontal",
        ))
        digest = hashlib.sha256("".join(d["raw_sha256"] for d in days).encode()).hexdigest()
        fingerprint[res.symbol] = {"days": len(days), "first": days[0]["day"], "last": days[-1]["day"], "sha256": digest}
        progress(f"{res.symbol}: {trades.height} trades -> {bars.height} bars, {feats.width} features")
    panel = pl.concat(parts).filter(pl.col("ts") < end).sort("ts", "symbol")
    feature_cols = [c for c in panel.columns if featureset.SEPARATOR in c]
    return panel, feature_cols, fingerprint


def _between(frame: pl.DataFrame, start, end) -> pl.DataFrame:
    return frame.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


def _fit_folds(rows: pl.DataFrame, feature_cols: list[str], label: Label, folds: list[Fold],
               params: dict) -> tuple[pl.DataFrame, list[dict]]:
    predictions, fold_records = [], []
    for i, fold in enumerate(folds, 1):
        train = _between(rows, fold.train_start, fold.train_end)
        val = _between(rows, fold.val_start, fold.val_end)
        if train.height < MIN_TRAIN_ROWS or val.is_empty():
            raise TradingError(
                f"fold {i}: {train.height} training rows and {val.height} validation rows; "
                f"need at least {MIN_TRAIN_ROWS} and 1. Check data coverage."
            )
        model = models.build_estimator(params, label.target)
        model.fit(train.select(feature_cols).to_numpy(), train[label.name].to_numpy())
        y_val = val[label.name].to_numpy()
        pred = np.asarray(models.predict(model, val.select(feature_cols).to_numpy(), label.target), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise TradingError(f"fold {i}: the model produced non-finite predictions")
        predictions.append(val.select("ts", "symbol", label.name).with_columns(
            pl.lit(i).alias("fold"), pl.Series("prediction", pred),
        ))
        fold_records.append({
            "fold": i,
            "train_start": fold.train_start.isoformat(), "train_end": fold.train_end.isoformat(),
            "val_start": fold.val_start.isoformat(), "val_end": fold.val_end.isoformat(),
            "n_train": train.height, "n_val": val.height,
            "metrics": diagnostics.model_metrics(y_val, pred, label.target),
        })
    return pl.concat(predictions), fold_records


def _summarize(oos: pl.DataFrame, label: Label, exp: Experiment, fold_records: list[dict]) -> tuple[dict, dict]:
    mode = diagnostics.ic_mode(oos)
    series, period = diagnostics.ic_series(oos, label.name, exp.cadence, mode)
    signal = diagnostics.signal_stats(series, mode, period, label.horizon).as_dict()
    keys = fold_records[0]["metrics"].keys()
    model = {k: float(np.nanmean([f["metrics"][k] for f in fold_records])) for k in keys}
    return signal, model


def run_experiment(path: str | Path, *, labels: list[str] | None = None, configs: list[str] | None = None,
                   rerun: bool = False, progress: Callable[[str], None] = print) -> RunReport:
    exp = load_experiment(path)
    setup = exp.setup
    known = {label.name for label in exp.labels}
    unknown = sorted(set(labels or []) - known)
    if unknown:
        raise TradingError(f"unknown labels {unknown}; this setup declares {sorted(known)}")
    selected = [label for label in exp.labels if not labels or label.name in labels]

    module = featureset.load_module(exp.path)
    progress(f"loading data [{exp.data_start:%Y-%m-%d}, {exp.holdout[0]:%Y-%m-%d}) - the holdout stays sealed")
    panel, feature_cols, fingerprint = _build_panel(exp, module, progress)

    log = RunLog(exp.path)
    report = RunReport(exp)
    identity_base = {
        "engine": __version__,
        "setup": {k: setup[k] for k in IDENTITY_SECTIONS},
        "features_sha256": featureset.module_digest(exp.path),
        "data": fingerprint,
    }
    for label in selected:
        rows = panel.drop_nulls([*feature_cols, label.name])
        report.dropped_rows[label.name] = panel.height - rows.height
        menu = yaml.safe_load((exp.path / "training" / f"{label.name}.yaml").read_text(encoding="utf-8"))
        for model_family, family_configs in menu["models"].items():
            for config_name, params in family_configs.items():
                if configs and config_name not in configs:
                    continue
                identity = {**identity_base, "label": label.name, "model_family": model_family,
                            "config_name": config_name, "params": params}
                rid = run_id(identity)
                meta = {"strategy_id": exp.strategy_id, "setup_version": setup["setup_version"], "label": label.name,
                        "model_family": model_family, "config_name": config_name, "params": params}
                trial = TrialResult(label.name, model_family, config_name, rid, "ok")
                existing = log.get(rid)
                if existing is not None and existing["status"] == "ok" and not rerun:
                    trial.reused = True
                    trial.signal = json.loads(existing["signal_json"])
                    trial.model = json.loads(existing["model_json"])
                    report.trials.append(trial)
                    progress(f"{label.name} / {config_name}: reused run {rid}")
                    continue
                progress(f"{label.name} / {config_name}: training {len(exp.folds_by_label[label.name])} folds")
                try:
                    oos, fold_records = _fit_folds(rows, feature_cols, label, exp.folds_by_label[label.name], params)
                    trial.signal, trial.model = _summarize(oos, label, exp, fold_records)
                    trial.n_oos = oos.height
                    log.record(rid, meta, identity, "ok", signal=trial.signal, model=trial.model,
                               folds=fold_records, predictions=oos)
                except Exception as e:  # noqa: BLE001 - every failure is a logged trial, not a crash
                    trial.status, trial.error = "failed", f"{type(e).__name__}: {e}"
                    log.record(rid, meta, identity, "failed", error=trial.error)
                report.trials.append(trial)
    return report
