"""Run log: every run, including failed ones, content-addressed by its configuration (ML4T ch06.7,
case_studies/RUN_LOG.md in the book's repository).

    experiments/<strategy_id>/run_log/
        registry.db                  SQLite index: runs and per-fold results
        predictions/<run_id>.parquet out-of-sample predictions of a successful run

A run's identity is the first 12 hex characters of the SHA-256 of its canonical configuration:
setup, label, model config, feature code, data fingerprint and engine version. Same identity,
same result: an existing successful run is reused instead of being trained again.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    setup_version TEXT NOT NULL,
    label TEXT NOT NULL,
    model_family TEXT NOT NULL,
    config_name TEXT NOT NULL,
    params_json TEXT NOT NULL,
    identity_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    signal_json TEXT,
    model_json TEXT
);
CREATE TABLE IF NOT EXISTS folds (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    fold INTEGER NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    val_start TEXT NOT NULL,
    val_end TEXT NOT NULL,
    n_train INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY (run_id, fold)
);
"""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_id(identity: dict) -> str:
    return hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()[:12]


class RunLog:
    def __init__(self, exp_dir: Path):
        self.root = exp_dir / "run_log"
        self.predictions_dir = self.root / "predictions"
        self.db_path = self.root / "registry.db"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def get(self, rid: str) -> sqlite3.Row | None:
        with closing(self._connect()) as db:
            return db.execute("SELECT * FROM runs WHERE run_id = ?", (rid,)).fetchone()

    def folds(self, rid: str) -> list[sqlite3.Row]:
        with closing(self._connect()) as db:
            return db.execute("SELECT * FROM folds WHERE run_id = ? ORDER BY fold", (rid,)).fetchall()

    def count(self, strategy_id: str) -> dict[str, int]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS n FROM runs WHERE strategy_id = ? GROUP BY status", (strategy_id,)
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def record(self, rid: str, meta: dict, identity: dict, status: str, *, error: str | None = None,
               signal: dict | None = None, model: dict | None = None, folds: list[dict] | None = None,
               predictions: pl.DataFrame | None = None) -> None:
        """Insert or replace a run. Predictions are written before the index row, so a run marked ok
        always has its artifact."""
        if predictions is not None:
            tmp = self.predictions_dir / f"{rid}.parquet.part"
            predictions.write_parquet(tmp)
            tmp.replace(self.predictions_dir / f"{rid}.parquet")
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM folds WHERE run_id = ?", (rid,))
            db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rid, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    meta["strategy_id"], meta["setup_version"], meta["label"], meta["model_family"],
                    meta["config_name"], canonical(meta["params"]), canonical(identity), status, error,
                    canonical(signal) if signal is not None else None,
                    canonical(model) if model is not None else None,
                ),
            )
            for f in folds or []:
                db.execute(
                    "INSERT INTO folds VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, f["fold"], f["train_start"], f["train_end"], f["val_start"], f["val_end"],
                     f["n_train"], f["n_val"], canonical(f["metrics"])),
                )
