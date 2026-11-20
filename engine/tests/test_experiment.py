import shutil
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from trading_engine.cli import main
from trading_engine.core.errors import TradingError
from trading_engine.experiment import load_experiment

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates" / "experiment"

SYMBOLS = ["AAAUSDT", "BBBUSDT"]  # synthetic: the plugin declares no assets
STRATEGY_ID = "synthetic_setup"

FILLED = {
    "strategy_id": STRATEGY_ID,
    "strategy_map": {
        "family": "price_based",
        "edge_source": "risk_compensation",
        "pnl_decomposition": ["price_pnl", "fees", "slippage"],
        "failure_modes": ["first test failure mode", "second test failure mode"],
    },
    "universe": {"symbols": SYMBOLS, "market": "crypto_perp_usdt", "requires": ["trades", "volume"],
                 "eligibility_rule": "always eligible"},
    "decision": {"cadence": "1h", "bar_type": "time", "snapshot": "bar_close", "execution_delay": "next_bar_open"},
    "mapping": {"position_state_space": "long_short", "entry_logic": "rank_top_k", "sizing": "fixed_fractional"},
    "costs": {"fee_maker_bps": 2, "fee_taker_bps": 5, "slippage_model": "estimated_half_spread",
              "sensitivity_bps": [0, 1, 2, 5, 10, 20, 50], "funding_modeled": False},
    "labels": [{"name": "fwd_ret_1h", "horizon": "1h", "target": "return", "primary": True},
               {"name": "fwd_ret_8h", "horizon": "8h", "target": "return"}],
    "features": {"families": [{"name": "test_family", "role": "signal", "hypothesis": "test hypothesis",
                               "inputs": "bars", "lookback": 24, "lag": 0, "failure_mode": "test failure"}]},
    "evaluation": {"protocol": "walk_forward", "n_splits": 4, "train_size": "365D", "val_size": "90D",
                   "data_start": "2023-01-01", "holdout": {"start": "2025-07-01", "end": "2026-01-01"}},
}
MENUS = {
    "fwd_ret_1h": {"linear": {"baseline": {"estimator": "ridge", "alpha": 1.0}},
                   "gbm": {"gbm_small": {"estimator": "lightgbm", "num_leaves": 15}}},
    "fwd_ret_8h": {"linear": {"baseline": {"estimator": "ridge", "alpha": 1.0}}},
}
FEATURES = 'import polars as pl\n\n\ndef test_family(bars):\n    return bars.select(pl.col("close").pct_change().alias("ret"))\n'


@pytest.fixture(autouse=True)
def declared_assets(isolated_env):
    (isolated_env / "catalog").mkdir()
    assets = {s: {"market": "crypto_perp_usdt", "source": "binance_um_futures", "streams": ["trades"]} for s in SYMBOLS}
    (isolated_env / "catalog" / "assets.yaml").write_text(yaml.safe_dump(assets), encoding="utf-8")


def _fill_template(dst: Path) -> Path:
    shutil.copytree(TEMPLATE_DIR, dst)
    setup = yaml.safe_load((dst / "setup.yaml").read_text(encoding="utf-8"))
    setup.update(FILLED)
    _write_setup(dst, setup)
    (dst / "training" / "CHANGE_ME.yaml").unlink()
    for label, models in MENUS.items():
        (dst / "training" / f"{label}.yaml").write_text(
            yaml.safe_dump({"label": label, "models": models}), encoding="utf-8")
    research = (dst / "research.md").read_text(encoding="utf-8")
    for heading in ("Hypothesis", "Economic mechanism", "Pre-registration", "Success criteria", "Kill criteria"):
        start = research.index(f"## {heading}\n") + len(f"## {heading}\n")
        end = research.index("\n", start)
        research = research[:start] + f"Filled {heading.lower()}." + research[end:]
    (dst / "research.md").write_text(research, encoding="utf-8")
    (dst / "features.py").write_text(FEATURES, encoding="utf-8")
    return dst


def _write_setup(exp_dir: Path, setup: dict) -> None:
    (exp_dir / "setup.yaml").write_text(yaml.safe_dump(setup, sort_keys=False), encoding="utf-8")


def _setup(exp_dir: Path) -> dict:
    return yaml.safe_load((exp_dir / "setup.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def exp_dir(tmp_path):
    return _fill_template(tmp_path / "synthetic_setup")


def test_raw_template_is_rejected(tmp_path):
    shutil.copytree(TEMPLATE_DIR, tmp_path / "raw")
    with pytest.raises(TradingError, match="placeholders"):
        load_experiment(tmp_path / "raw")


def test_filled_template_passes_and_counts_trials(exp_dir):
    exp = load_experiment(exp_dir)
    assert exp.trials_by_label == {"fwd_ret_1h": 2, "fwd_ret_8h": 1}
    assert exp.total_trials == 3
    assert [label.name for label in exp.labels if label.primary] == ["fwd_ret_1h"]
    assert any("funding" in w for w in exp.warnings)


def test_folds_follow_the_book_geometry(exp_dir):
    exp = load_experiment(exp_dir)
    holdout_start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    horizon = timedelta(hours=8)
    folds = exp.folds_by_label["fwd_ret_8h"]
    assert len(folds) == 4
    assert folds[-1].val_end == holdout_start - horizon
    for fold in folds:
        assert fold.val_start - fold.train_end == horizon
        assert fold.val_start - fold.train_start == timedelta(days=365)
    for fold in folds[:-1]:
        assert fold.val_end - fold.val_start == timedelta(days=90)
    for earlier, later in pairwise(folds):
        assert earlier.val_end == later.val_start
    short = exp.folds_by_label["fwd_ret_1h"]
    assert [f.val_start for f in short] == [f.val_start for f in folds]


def test_missing_feature_function_rejected(exp_dir):
    (exp_dir / "features.py").write_text("def other(bars):\n    return bars\n", encoding="utf-8")
    with pytest.raises(TradingError, match="no function for the declared families"):
        load_experiment(exp_dir)


def test_estimator_must_match_label_target(exp_dir):
    menu = exp_dir / "training" / "fwd_ret_8h.yaml"
    menu.write_text("label: fwd_ret_8h\nmodels:\n  linear:\n    l1: {estimator: logistic}\n", encoding="utf-8")
    with pytest.raises(TradingError, match="serves direction labels"):
        load_experiment(exp_dir)


def test_insufficient_history_is_rejected(exp_dir):
    setup = _setup(exp_dir)
    setup["evaluation"]["data_start"] = "2024-06-01"
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match="needs data from"):
        load_experiment(exp_dir)


@pytest.mark.parametrize("path, value, message", [
    (("decision", "execution_delay"), "same_bar", "no same-bar fills"),
    (("decision", "cadence"), "7m", "timeframes.yaml"),
    (("metrics", "selection"), "net_sharpe", "model or signal layer"),
    (("costs", "slippage_model"), "quoted_half_spread", "needs quotes"),
    (("costs", "sensitivity_bps"), [5, 10], "starts at 0"),
    (("setup_version",), "1", "v1, v2"),
    (("strategy_map", "family"), "momentum", "strategy_map.family"),
])
def test_invalid_setup_fields(exp_dir, path, value, message):
    setup = _setup(exp_dir)
    node = setup
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match=message):
        load_experiment(exp_dir)


def test_volume_is_mandatory(exp_dir):
    setup = _setup(exp_dir)
    setup["universe"]["requires"] = ["trades"]
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match="volume"):
        load_experiment(exp_dir)


def test_label_horizon_must_be_multiple_of_cadence(exp_dir):
    setup = _setup(exp_dir)
    setup["labels"][1]["horizon"] = "90min"
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match="multiple of the decision cadence"):
        load_experiment(exp_dir)


def test_exactly_one_primary_label(exp_dir):
    setup = _setup(exp_dir)
    setup["labels"][1]["primary"] = True
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match="exactly one label"):
        load_experiment(exp_dir)


def test_signal_family_required(exp_dir):
    setup = _setup(exp_dir)
    setup["features"]["families"][0]["role"] = "state"
    _write_setup(exp_dir, setup)
    with pytest.raises(TradingError, match="role: signal"):
        load_experiment(exp_dir)


def test_linear_baseline_required(exp_dir):
    menu = exp_dir / "training" / "fwd_ret_8h.yaml"
    menu.write_text("label: fwd_ret_8h\nmodels:\n  gbm:\n    g1: {estimator: lightgbm}\n", encoding="utf-8")
    with pytest.raises(TradingError, match="baseline"):
        load_experiment(exp_dir)


def test_missing_training_menu_rejected(exp_dir):
    (exp_dir / "training" / "fwd_ret_8h.yaml").unlink()
    with pytest.raises(TradingError, match="missing"):
        load_experiment(exp_dir)


def test_unfilled_research_rejected(exp_dir):
    shutil.copy(TEMPLATE_DIR / "research.md", exp_dir / "research.md")
    with pytest.raises(TradingError, match="Hypothesis"):
        load_experiment(exp_dir)


def test_cli_prints_fold_plan(exp_dir, capsys):
    assert main(["check-experiment", str(exp_dir)]) == 0
    out = capsys.readouterr().out
    assert "OK synthetic_setup v1" in out
    assert "sealed holdout [2025-07-01, 2026-01-01)" in out
    assert "search accounting: 3 trials" in out
    assert out.count("fold ") == 8


def test_fold_plan_matches_book_walk_forward_cv(exp_dir):
    """Parity with utils/cv_splits.py in the book's repository: WalkForwardCV backward, rolling,
    plus the purge of validation rows whose label reaches the holdout."""
    import numpy as np
    import pandas as pd
    from ml4t.diagnostic.splitters import WalkForwardCV
    from ml4t.diagnostic.splitters.config import WalkForwardConfig

    exp = load_experiment(exp_dir)
    holdout = pd.Timestamp(exp.holdout[0])
    index = pd.date_range(pd.Timestamp(exp.data_start), holdout - pd.Timedelta("1h"), freq="1h")
    frame = pd.DataFrame({"_d": np.zeros(len(index), dtype=np.int8)}, index=index)
    for label in exp.labels:
        horizon = f"{int(label.horizon.total_seconds() // 3600)}h"
        config = WalkForwardConfig(
            n_splits=4, train_size="365D", val_size="90D", holdout_start=str(holdout.date()),
            holdout_end="2026-01-01", label_horizon=horizon, calendar_id=None, fold_direction="backward",
        )
        cv = WalkForwardCV(config=config)
        cv.expanding = False
        for (train_idx, val_idx), fold in zip(cv.split(frame), exp.folds_by_label[label.name], strict=True):
            val_idx = val_idx[index[val_idx] < holdout - pd.Timedelta(horizon)]
            train, val = index[train_idx], index[val_idx]
            ours_train = index[(index >= fold.train_start) & (index < fold.train_end)]
            ours_val = index[(index >= fold.val_start) & (index < fold.val_end)]
            assert train.equals(ours_train), label.name
            assert val.equals(ours_val), label.name
