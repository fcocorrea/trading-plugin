"""Experiment definition and validation, following the ML4T research framework (ch06-ch08).

An experiment is a folder that mirrors a case study in the book's repository
(case_studies/<id>/config/ and run_log/):

    experiments/<strategy_id>/
        setup.yaml            versioned trading setup: the invariants of the research program
        training/<label>.yaml model menu for one label; every entry is a trial
        research.md           strategy map, pre-registration and decision log
        run_log/              every run, including failed and discarded ones (pending)

Validation happens before any data is loaded. A setup that fails here does not run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from . import catalog, featureset, models
from .core.errors import TradingError
from .core.timeutil import parse_duration

PLACEHOLDER = "CHANGE_ME"
SETUP_VERSION_RE = re.compile(r"^v[1-9]\d*$")
BAR_TYPES = ("time", "tick", "volume", "dollar")
EXECUTION_DELAYS = ("next_bar_open", "next_quote")
STATE_SPACES = ("long_only", "short_only", "long_short")
LABEL_TARGETS = ("return", "direction")
FEATURE_ROLES = ("signal", "state")
PROTOCOLS = ("walk_forward",)
SLIPPAGE_MODELS = ("quoted_half_spread", "estimated_half_spread", "fixed_bps")
# Model and signal diagnostics only: strategy outcomes as a selection target overfit the simulator.
SELECTION_METRICS = ("rank_ic", "ic_ir", "mse", "mae", "log_loss", "auc", "hit_rate")
RESEARCH_SECTIONS = (
    "Hypothesis",
    "Economic mechanism",
    "Pre-registration",
    "Success criteria",
    "Kill criteria",
)


@dataclass(frozen=True)
class Label:
    name: str
    horizon: timedelta
    target: str
    primary: bool


@dataclass(frozen=True)
class Fold:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime


@dataclass
class Experiment:
    path: Path
    setup: dict
    labels: list[Label]
    cadence: timedelta
    holdout: tuple[datetime, datetime]
    data_start: datetime
    folds_by_label: dict[str, list[Fold]]
    trials_by_label: dict[str, int]
    max_feature_lookback: timedelta
    warnings: list[str] = field(default_factory=list)

    @property
    def strategy_id(self) -> str:
        return str(self.setup["strategy_id"])

    @property
    def total_trials(self) -> int:
        return sum(self.trials_by_label.values())


def _utc(d: Any, where: str) -> datetime:
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    if isinstance(d, date):
        return datetime.combine(d, time(), tzinfo=timezone.utc)
    try:
        return datetime.combine(date.fromisoformat(str(d)), time(), tzinfo=timezone.utc)
    except ValueError as e:
        raise TradingError(f"{where}: invalid date '{d}', use YYYY-MM-DD") from e


def _section(cfg: dict, key: str, where: str = "setup.yaml") -> dict:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise TradingError(f"{where}: '{key}' must be a mapping")
    return value


def _require(cfg: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if cfg.get(k) in (None, "", [])]
    if missing:
        raise TradingError(f"{where}: missing required fields {missing}")


def _one_of(value: Any, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise TradingError(f"{where}: '{value}' is not one of {allowed}")
    return str(value)


def _placeholders(node: Any, prefix: str = "") -> list[str]:
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _placeholders(v, f"{prefix}.{k}" if prefix else str(k))]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _placeholders(v, f"{prefix}[{i}]")]
    return [prefix] if PLACEHOLDER in str(node) else []


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise TradingError(f"missing {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TradingError(f"{path.name}: top level must be a mapping")
    leftovers = _placeholders(data)
    if leftovers:
        raise TradingError(f"{path.name}: template placeholders still present at {leftovers}")
    return data


def _check_universe(setup: dict) -> str:
    universe = _section(setup, "universe")
    _require(universe, ("symbols", "market", "requires", "eligibility_rule"), "universe")
    symbols, requires = universe["symbols"], universe["requires"]
    if not isinstance(symbols, list) or not isinstance(requires, list):
        raise TradingError("universe: 'symbols' and 'requires' must be lists")
    if "volume" not in requires:
        raise TradingError("universe.requires must include 'volume': no traded size, no experiment")
    for symbol in symbols:
        resolved = catalog.resolve(str(symbol), requires=requires)
        if resolved.market_id != universe["market"]:
            raise TradingError(
                f"universe: {symbol} belongs to market {resolved.market_id}, not {universe['market']}"
            )
    return str(universe["market"])


def _check_strategy_map(setup: dict) -> None:
    smap = _section(setup, "strategy_map")
    _require(smap, ("family", "edge_source", "pnl_decomposition", "failure_modes"), "strategy_map")
    reference = catalog.load_strategy_map()
    _one_of(smap["family"], tuple(reference["families"]), "strategy_map.family")
    _one_of(smap["edge_source"], tuple(reference["edge_sources"]), "strategy_map.edge_source")
    if not isinstance(smap["failure_modes"], list) or len(smap["failure_modes"]) < 2:
        raise TradingError("strategy_map.failure_modes: name the top 2-3 ways this strategy fails")


def _check_decision(setup: dict) -> timedelta:
    decision = _section(setup, "decision")
    _require(decision, ("cadence", "bar_type", "snapshot", "execution_delay"), "decision")
    timeframes = catalog.load_timeframes()
    cadence = str(decision["cadence"])
    if cadence not in timeframes:
        raise TradingError(f"decision.cadence '{cadence}' is not in timeframes.yaml {sorted(timeframes)}")
    _one_of(decision["bar_type"], BAR_TYPES, "decision.bar_type")
    _one_of(decision["execution_delay"], EXECUTION_DELAYS, "decision.execution_delay (no same-bar fills)")
    return timedelta(minutes=int(timeframes[cadence]["minutes"]))


def _check_mapping(setup: dict, market_id: str) -> None:
    mapping = _section(setup, "mapping")
    _require(mapping, ("position_state_space", "entry_logic", "sizing"), "mapping")
    state_space = _one_of(mapping["position_state_space"], STATE_SPACES, "mapping.position_state_space")
    market = catalog.load_markets()[market_id]
    if state_space != "long_only" and market.get("short_allowed") is not True:
        raise TradingError(f"mapping: market {market_id} does not allow unrestricted shorting")


def _check_costs(setup: dict) -> None:
    costs = _section(setup, "costs")
    _require(costs, ("fee_maker_bps", "fee_taker_bps", "slippage_model", "sensitivity_bps"), "costs")
    for key in ("fee_maker_bps", "fee_taker_bps"):
        if not isinstance(costs[key], (int, float)) or costs[key] < 0:
            raise TradingError(f"costs.{key} must be a non-negative number")
    model = _one_of(costs["slippage_model"], SLIPPAGE_MODELS, "costs.slippage_model")
    requires = setup["universe"]["requires"]
    if model == "quoted_half_spread" and "quotes" not in requires:
        raise TradingError("costs: quoted_half_spread needs quotes; add it to universe.requires")
    if model == "fixed_bps" and not isinstance(costs.get("slippage_bps"), (int, float)):
        raise TradingError("costs: fixed_bps needs a numeric costs.slippage_bps")
    grid = costs["sensitivity_bps"]
    if not isinstance(grid, list) or len(grid) < 2 or min(grid) != 0:
        raise TradingError("costs.sensitivity_bps: a cost grid that starts at 0, e.g. [0, 2, 5, 10, 20]")


def _check_labels(setup: dict, cadence: timedelta) -> list[Label]:
    raw = setup.get("labels")
    if not isinstance(raw, list) or not raw:
        raise TradingError("labels: declare at least one label")
    labels = []
    for i, item in enumerate(raw):
        where = f"labels[{i}]"
        if not isinstance(item, dict):
            raise TradingError(f"{where} must be a mapping")
        _require(item, ("name", "horizon", "target"), where)
        horizon = parse_duration(item["horizon"])
        if horizon < cadence or horizon % cadence:
            raise TradingError(f"{where}: horizon {item['horizon']} must be a multiple of the decision cadence")
        labels.append(Label(
            name=str(item["name"]),
            horizon=horizon,
            target=_one_of(item["target"], LABEL_TARGETS, f"{where}.target"),
            primary=bool(item.get("primary", False)),
        ))
    if sum(label.primary for label in labels) != 1:
        raise TradingError("labels: exactly one label must be primary: true")
    names = [label.name for label in labels]
    if len(set(names)) != len(names):
        raise TradingError(f"labels: duplicated names {names}")
    return labels


def _check_features(setup: dict, cadence: timedelta) -> timedelta:
    families = _section(setup, "features").get("families")
    if not isinstance(families, list) or not families:
        raise TradingError("features.families: declare at least one feature family")
    keys = ("name", "role", "hypothesis", "inputs", "lookback", "lag", "failure_mode")
    for i, fam in enumerate(families):
        where = f"features.families[{i}]"
        if not isinstance(fam, dict):
            raise TradingError(f"{where} must be a mapping")
        _require(fam, keys, where)
        _one_of(fam["role"], FEATURE_ROLES, f"{where}.role")
        for key in ("lookback", "lag"):
            if not isinstance(fam[key], int) or fam[key] < 0:
                raise TradingError(f"{where}.{key} must be a non-negative integer number of bars")
    if not any(fam["role"] == "signal" for fam in families):
        raise TradingError("features.families: at least one family must have role: signal")
    return max(fam["lookback"] + fam["lag"] for fam in families) * cadence


def _check_metrics(setup: dict) -> None:
    metrics = _section(setup, "metrics")
    _require(metrics, ("selection", "report"), "metrics")
    _one_of(metrics["selection"], SELECTION_METRICS, "metrics.selection (model or signal layer only)")


def plan_folds(evaluation: dict, holdout_start: datetime, horizon: timedelta) -> list[Fold]:
    """Rolling walk-forward folds, oldest first, matching utils/cv_splits.py in the book's repository
    (WalkForwardCV with fold_direction='backward', expanding=False, plus its holdout purge):

    - validation windows of val_size tile backward from the holdout start, identical for every label;
    - a training window spans the train_size before its validation window, minus the rows whose
      label would reach into validation (purge = label horizon);
    - the last validation window drops the rows whose label would reach into the holdout.
    """
    train_size = parse_duration(evaluation["train_size"])
    val_size = parse_duration(evaluation["val_size"])
    n_splits = int(evaluation["n_splits"])
    folds = []
    for k in range(n_splits, 0, -1):
        val_end = holdout_start - (k - 1) * val_size
        val_start = val_end - val_size
        if k == 1:
            val_end = holdout_start - horizon
        folds.append(Fold(val_start - train_size, val_start - horizon, val_start, val_end))
    return folds


def _check_evaluation(setup: dict, labels: list[Label], lookback: timedelta):
    evaluation = _section(setup, "evaluation")
    _require(evaluation, ("protocol", "n_splits", "train_size", "val_size", "data_start", "holdout"), "evaluation")
    _one_of(evaluation["protocol"], PROTOCOLS, "evaluation.protocol")
    if not isinstance(evaluation["n_splits"], int) or evaluation["n_splits"] < 1:
        raise TradingError("evaluation.n_splits must be a positive integer")
    holdout = evaluation["holdout"]
    if not isinstance(holdout, dict):
        raise TradingError("evaluation.holdout must be {start, end}")
    _require(holdout, ("start", "end"), "evaluation.holdout")
    h_start = _utc(holdout["start"], "evaluation.holdout.start")
    h_end = _utc(holdout["end"], "evaluation.holdout.end")
    if h_start >= h_end:
        raise TradingError("evaluation.holdout: start must be before end; the period is [start, end)")
    data_start = _utc(evaluation["data_start"], "evaluation.data_start")

    folds_by_label = {}
    for label in labels:
        folds = plan_folds(evaluation, h_start, label.horizon)
        if any(f.val_end <= f.val_start or f.train_end <= f.train_start for f in folds):
            raise TradingError(
                f"evaluation: label {label.name} leaves an empty training or validation window; "
                "its horizon is too long for train_size or val_size"
            )
        needed = folds[0].train_start - lookback
        if needed < data_start:
            raise TradingError(
                f"evaluation: label {label.name} needs data from {needed:%Y-%m-%d %H:%M} "
                f"(first training window plus {lookback} of feature warm-up), but data_start is "
                f"{data_start:%Y-%m-%d}. Move data_start back, or reduce n_splits, train_size or val_size."
            )
        folds_by_label[label.name] = folds
    return (h_start, h_end), data_start, folds_by_label


def _check_training(exp_dir: Path, labels: list[Label]) -> dict[str, int]:
    trials = {}
    for label in labels:
        path = exp_dir / "training" / f"{label.name}.yaml"
        menu = _load_yaml(path)
        if menu.get("label") != label.name:
            raise TradingError(f"{path.name}: 'label' must be {label.name}")
        menu_models = menu.get("models")
        if not isinstance(menu_models, dict) or not menu_models:
            raise TradingError(f"{path.name}: 'models' must map model families to named configs")
        if not menu_models.get("linear"):
            raise TradingError(f"{path.name}: needs a 'linear' family; it is the baseline every model must beat")
        count = 0
        for family, configs in menu_models.items():
            if not isinstance(configs, dict) or not configs:
                raise TradingError(f"{path.name}: models.{family} must map config names to parameters")
            for name, params in configs.items():
                where = f"{path.name}: models.{family}.{name}"
                if not isinstance(params, dict) or "estimator" not in params:
                    raise TradingError(f"{where} needs an 'estimator'")
                spec = models.ESTIMATORS.get(params["estimator"])
                if spec is None:
                    raise TradingError(f"{where}: unknown estimator '{params['estimator']}'; available: {sorted(models.ESTIMATORS)}")
                if spec.target != label.target:
                    raise TradingError(f"{where}: estimator '{params['estimator']}' serves {spec.target} labels, not {label.target}")
                if family == "linear" and params["estimator"] not in models.LINEAR_ESTIMATORS:
                    raise TradingError(f"{where}: the linear baseline must use one of {models.LINEAR_ESTIMATORS}")
                count += 1
        trials[label.name] = count
    return trials


def _check_research(exp_dir: Path) -> None:
    path = exp_dir / "research.md"
    if not path.is_file():
        raise TradingError(f"missing {path}: write the hypothesis before running anything")
    text = path.read_text(encoding="utf-8")
    for heading in RESEARCH_SECTIONS:
        match = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
        body = match.group(1).strip() if match else ""
        if not body or re.fullmatch(r"<[^>]*>", body):
            raise TradingError(f"research.md: section '## {heading}' is missing or still a placeholder")


def load_experiment(path: str | Path) -> Experiment:
    exp_dir = Path(path)
    if exp_dir.name == "setup.yaml":
        exp_dir = exp_dir.parent
    setup = _load_yaml(exp_dir / "setup.yaml")
    _require(setup, ("strategy_id", "setup_version"), "setup.yaml")
    if not SETUP_VERSION_RE.match(str(setup["setup_version"])):
        raise TradingError("setup_version must look like v1, v2, ...")

    market_id = _check_universe(setup)
    _check_strategy_map(setup)
    cadence = _check_decision(setup)
    _check_mapping(setup, market_id)
    _check_costs(setup)
    labels = _check_labels(setup, cadence)
    lookback = _check_features(setup, cadence)
    _check_metrics(setup)
    holdout, data_start, folds = _check_evaluation(setup, labels, lookback)
    trials = _check_training(exp_dir, labels)
    _check_research(exp_dir)
    featureset.check_families(featureset.load_module(exp_dir), setup["features"]["families"])

    exp = Experiment(exp_dir, setup, labels, cadence, holdout, data_start, folds, trials, lookback)
    market = catalog.load_markets()[market_id]
    if market.get("settlement") and not setup["costs"].get("funding_modeled", False):
        exp.warnings.append(
            f"market {market_id} has settlement '{market['settlement']}' that the engine does not model "
            "yet: results will exclude funding and must be labeled as such"
        )
    return exp
