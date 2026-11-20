from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from helpers import FEATURES_LOOKAHEAD, runlog_rows, write_assets, write_experiment, write_market
from trading_engine.bars import time_bars
from trading_engine.cli import main
from trading_engine.core.errors import DataValidationError
from trading_engine.experiment import Label
from trading_engine.featureset import LookAheadError
from trading_engine.labels import add_labels
from trading_engine.runner import run_experiment

SYMBOLS = ["TESTA", "TESTB"]
DATA_START, HOLDOUT = date(2024, 1, 1), date(2024, 3, 1)
quiet = lambda _msg: None  # noqa: E731


@pytest.fixture
def market(isolated_env):
    write_assets(isolated_env, SYMBOLS)
    write_market(SYMBOLS, DATA_START, HOLDOUT)  # nothing written for the holdout: reading it would fail
    return isolated_env


def test_planted_signal_is_found_and_logged(market):
    exp_dir = write_experiment(market / "experiments" / "planted", SYMBOLS)
    report = run_experiment(exp_dir, progress=quiet)

    by_config = {t.config_name: t for t in report.trials}
    assert {t.status for t in report.trials} == {"ok"}
    ridge = by_config["ridge_a1"]
    assert ridge.signal["mode"] == "time_series"
    assert ridge.signal["period"] == "1D"
    assert ridge.signal["mean_ic"] > 0.2
    assert ridge.signal["t_stat"] > 3
    assert by_config["logit"].model["auc"] > 0.6

    rows = runlog_rows(exp_dir)
    assert len(rows) == 3
    assert all((exp_dir / "run_log" / "predictions" / f"{r['run_id']}.parquet").exists() for r in rows)
    oos = pl.read_parquet(exp_dir / "run_log" / "predictions" / f"{ridge.run_id}.parquet")
    assert oos["ts"].max() < datetime(2024, 3, 1, tzinfo=timezone.utc) - timedelta(hours=1)
    assert sorted(oos["fold"].unique().to_list()) == [1, 2, 3]


def test_identical_runs_are_reused(market):
    exp_dir = write_experiment(market / "experiments" / "reuse", SYMBOLS)
    first = run_experiment(exp_dir, labels=["fwd_ret_1h"], configs=["ridge_a1"], progress=quiet)
    second = run_experiment(exp_dir, labels=["fwd_ret_1h"], configs=["ridge_a1"], progress=quiet)
    assert not first.trials[0].reused
    assert second.trials[0].reused
    assert second.trials[0].run_id == first.trials[0].run_id
    assert second.trials[0].signal == first.trials[0].signal


def test_lookahead_feature_is_refused(market):
    exp_dir = write_experiment(market / "experiments" / "peek", SYMBOLS, features=FEATURES_LOOKAHEAD)
    with pytest.raises(LookAheadError, match="flow__next_return"):
        run_experiment(exp_dir, progress=quiet)


def test_missing_data_is_refused(market):
    exp_dir = write_experiment(market / "experiments" / "gap", SYMBOLS, holdout_start="2024-03-10")
    with pytest.raises(DataValidationError, match="days missing"):
        run_experiment(exp_dir, progress=quiet)


def test_failed_trial_is_logged_not_raised(market):
    menu = {
        "fwd_ret_1h": {"linear": {"ridge_bad": {"estimator": "ridge", "alpha": -5.0}}},
        "fwd_dir_1h": {"linear": {"logit": {"estimator": "logistic"}}},
    }
    exp_dir = write_experiment(market / "experiments" / "bad", SYMBOLS, menu=menu)
    report = run_experiment(exp_dir, labels=["fwd_ret_1h"], progress=quiet)
    assert report.trials[0].status == "failed"
    assert runlog_rows(exp_dir)[0]["status"] == "failed"


def test_cross_sectional_mode_with_ten_symbols(isolated_env):
    symbols = [f"SYM{i:02d}" for i in range(10)]
    write_assets(isolated_env, symbols)
    write_market(symbols, DATA_START, HOLDOUT)
    exp_dir = write_experiment(isolated_env / "experiments" / "xs", symbols)
    report = run_experiment(exp_dir, labels=["fwd_ret_1h"], configs=["ridge_a1"], progress=quiet)
    assert report.trials[0].signal["mode"] == "cross_sectional"
    assert report.trials[0].signal["mean_ic"] > 0.2


def test_cli_run_prints_table(market, capsys):
    exp_dir = write_experiment(market / "experiments" / "cli", SYMBOLS)
    assert main(["run", str(exp_dir), "--label", "fwd_ret_1h", "--config", "ridge_a1"]) == 0
    out = capsys.readouterr().out
    assert "ridge_a1" in out and "ts/1D" in out
    assert "The sealed holdout was not read." in out


def test_time_bars_fill_gaps_and_split_flow():
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000)
    hour = 3_600_000_000
    trades = pl.DataFrame({
        "price": [100.0, 101.0, 103.0], "size": [1.0, 2.0, 1.0], "quote_volume": [100.0, 202.0, 103.0],
        "aggressor_side": ["buy", "sell", "buy"], "ts_exchange_us": [t0 + 1, t0 + 2, t0 + 2 * hour + 5],
    })
    bars = time_bars(trades, "X", timedelta(hours=1))
    assert bars["ts"].to_list() == [datetime(2024, 1, 1, h, tzinfo=timezone.utc) for h in (1, 2, 3)]
    assert bars["close"].to_list() == [101.0, 101.0, 103.0]
    assert bars["volume"].to_list() == [3.0, 0.0, 1.0]
    assert bars["buy_volume"].to_list() == [1.0, 0.0, 1.0]
    assert bars["sell_volume"].to_list() == [2.0, 0.0, 0.0]


def test_label_enters_at_next_open_and_exits_after_horizon():
    bars = pl.DataFrame({"open": [10.0, 11.0, 12.0, 15.0, 18.0]})
    labels = [Label("fwd_ret_2h", timedelta(hours=2), "return", True),
              Label("fwd_dir_2h", timedelta(hours=2), "direction", False)]
    out = add_labels(bars, labels, timedelta(hours=1), "next_bar_open")
    assert out["fwd_ret_2h"].to_list()[:2] == [pytest.approx(15 / 11 - 1), pytest.approx(18 / 12 - 1)]
    assert out["fwd_ret_2h"].to_list()[2:] == [None, None, None]
    assert out["fwd_dir_2h"].to_list()[:2] == [1, 1]


def test_no_planted_signal_finds_nothing(isolated_env):
    write_assets(isolated_env, SYMBOLS)
    write_market(SYMBOLS, DATA_START, HOLDOUT, beta=0.0)
    exp_dir = write_experiment(isolated_env / "experiments" / "null", SYMBOLS)
    report = run_experiment(exp_dir, labels=["fwd_ret_1h"], configs=["ridge_a1"], progress=quiet)
    stats = report.trials[0].signal
    assert abs(stats["mean_ic"]) < 0.05
    assert abs(stats["t_stat"]) < 3
