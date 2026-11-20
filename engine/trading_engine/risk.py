"""Personal risk profile: loading and validation (see skills/risk-management).

The profile lives in the workspace (risk_profile.yaml) because risk tolerance is personal. It holds
the simulated account, hard limits, portfolio kill switches and a trade-management configuration
that must stay within the limits. Every distance is a multiple of ATR, never a fixed amount of price.
Where a personal rule and Machine Learning for Trading differ, the book wins (ch17, ch19).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import yaml

from .core.errors import TradingError
from .core.paths import workspace_root
from .core.timeutil import parse_duration

PROFILE_FILE = "risk_profile.yaml"
VOLATILITY_MEASURES = ("atr",)
LADDER_ACTIONS = ("watch", "reduce_risk_50", "stop_new", "unwind", "terminate")
REINSTATEMENTS = ("pause_elapsed",)
ACCOUNT_SPECS = ("leverage", "margin_call_level_pct", "stop_out_level_pct")
SYMBOL_SPECS = ("contract_size", "min_lot", "lot_step", "max_lot", "swap_long", "swap_short")


@dataclass(frozen=True)
class RiskProfile:
    path: Path
    raw: dict
    risk_per_trade_pct: float
    stop_atr: float
    take_profit_atr: float
    reward_risk: float
    pause: timedelta
    initial_balance_usd: float
    warnings: list[str] = field(default_factory=list)

    @property
    def account(self) -> dict:
        return self.raw["account"]

    @property
    def limits(self) -> dict:
        return self.raw["limits"]

    @property
    def kill_switch(self) -> dict:
        return self.raw["kill_switch"]

    @property
    def management(self) -> dict:
        return self.raw["trade_management"]


def default_path() -> Path:
    return workspace_root() / PROFILE_FILE


def _mapping(node: dict, key: str, where: str) -> dict:
    value = node.get(key)
    if not isinstance(value, dict):
        raise TradingError(f"{where}.{key} must be a mapping")
    return value


def _number(node: dict, key: str, where: str, *, low: float = 0.0, high: float | None = None,
            strict_low: bool = True) -> float:
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TradingError(f"{where}.{key} must be a number")
    if (strict_low and value <= low) or (not strict_low and value < low) or (high is not None and value > high):
        bound = f"({'>' if strict_low else '>='} {low}{f', <= {high}' if high is not None else ''})"
        raise TradingError(f"{where}.{key} = {value} is out of range {bound}")
    return float(value)


def _enabled(node: dict, where: str) -> bool:
    if not isinstance(node.get("enabled"), bool):
        raise TradingError(f"{where}.enabled must be true or false")
    return node["enabled"]


def _check_account(account: dict, warnings: list[str]) -> float:
    for key in ("broker", "type", "platform", "currency"):
        if not isinstance(account.get(key), str) or not account[key]:
            raise TradingError(f"account.{key} is required")
    per_usd = _number(account, "usc_per_usd", "account") if account["currency"] == "USC" else 1.0
    balance = _number(account, "initial_balance", "account")
    pending = [k for k in ACCOUNT_SPECS if account.get(k) is None]
    for key in ACCOUNT_SPECS:
        if account.get(key) is not None:
            _number(account, key, "account", strict_low=key != "stop_out_level_pct")
    symbols = account.get("symbols")
    if not isinstance(symbols, dict):
        raise TradingError("account.symbols must map each traded symbol to its contract specification")
    if not symbols:
        warnings.append("account.symbols is empty: add the contract specification of each symbol before simulating it")
    for symbol, spec in symbols.items():
        where = f"account.symbols.{symbol}"
        if not isinstance(spec, dict):
            raise TradingError(f"{where} must be a mapping")
        for key in SYMBOL_SPECS:
            if spec.get(key) is None:
                pending.append(f"{symbol}.{key}")
            elif key.startswith("swap"):
                if isinstance(spec[key], bool) or not isinstance(spec[key], (int, float)):
                    raise TradingError(f"{where}.{key} must be a number")
            else:
                _number(spec, key, where)
        if all(isinstance(spec.get(k), (int, float)) for k in ("min_lot", "max_lot")) and spec["min_lot"] > spec["max_lot"]:
            raise TradingError(f"{where}: min_lot exceeds max_lot")
    if pending:
        warnings.append(
            "broker conditions not confirmed yet (needed before simulating trades): " + ", ".join(pending)
            + ". Read them in MT5: Market Watch > Specification, and the account page."
        )
    return balance / per_usd


def _check_limits(limits: dict) -> tuple[float, float, float]:
    max_risk = _number(limits, "max_risk_per_trade_pct", "limits", high=100.0)
    total = _number(limits, "max_total_risk_pct", "limits", high=100.0)
    if total < max_risk:
        raise TradingError("limits.max_total_risk_pct must be at least max_risk_per_trade_pct")
    _number(limits, "max_position_pct", "limits")
    rr = _mapping(limits, "reward_risk", "limits")
    rr_min = _number(rr, "min", "limits.reward_risk")
    rr_max = _number(rr, "max", "limits.reward_risk")
    if rr_min > rr_max:
        raise TradingError("limits.reward_risk: min must not exceed max")

    rsi = _mapping(limits, "rsi_filter", "limits")
    period = rsi.get("period")
    if not isinstance(period, int) or period < 2:
        raise TradingError("limits.rsi_filter.period must be an integer >= 2")
    low = _number(rsi, "no_sell_below", "limits.rsi_filter", high=100.0)
    high = _number(rsi, "no_buy_above", "limits.rsi_filter", high=100.0)
    if not low < high:
        raise TradingError("limits.rsi_filter: no_sell_below must be lower than no_buy_above")
    return max_risk, rr_min, rr_max


def _check_kill_switch(ks: dict) -> timedelta:
    streak = _mapping(ks, "consecutive_losses", "kill_switch")
    losses = streak.get("losses")
    if not isinstance(losses, int) or losses < 1:
        raise TradingError("kill_switch.consecutive_losses.losses must be a positive integer")
    if streak.get("reinstatement") not in REINSTATEMENTS:
        raise TradingError(f"kill_switch.consecutive_losses.reinstatement must be one of {REINSTATEMENTS}")
    pause = parse_duration(streak.get("pause"))

    ladder = ks.get("drawdown_ladder")
    if not isinstance(ladder, list) or not ladder:
        raise TradingError("kill_switch.drawdown_ladder must list graduated drawdown levels (ch19, Table 19.7)")
    previous = 0.0
    for i, level in enumerate(ladder):
        where = f"kill_switch.drawdown_ladder[{i}]"
        if not isinstance(level, dict):
            raise TradingError(f"{where} must be a mapping")
        dd = _number(level, "drawdown_pct", where, high=100.0)
        if dd <= previous:
            raise TradingError("kill_switch.drawdown_ladder: drawdown_pct must increase level by level")
        if level.get("action") not in LADDER_ACTIONS:
            raise TradingError(f"{where}.action must be one of {LADDER_ACTIONS}")
        previous = dd
    if not any(level["action"] in ("stop_new", "unwind", "terminate") for level in ladder):
        raise TradingError("kill_switch.drawdown_ladder needs a level that stops new positions")
    return pause


def _check_management(tm: dict, max_risk: float, rr_min: float, rr_max: float) -> tuple[float, float, float, float]:
    risk = _number(tm, "risk_per_trade_pct", "trade_management")
    if risk > max_risk:
        raise TradingError(
            f"trade_management.risk_per_trade_pct = {risk} exceeds the personal limit of {max_risk}% of equity"
        )
    stop = _number(_mapping(tm, "stop_loss", "trade_management"), "atr_multiple", "trade_management.stop_loss")
    target = _number(_mapping(tm, "take_profit", "trade_management"), "atr_multiple", "trade_management.take_profit")
    rr = target / stop
    if not rr_min <= rr <= rr_max:
        raise TradingError(
            f"reward/risk = take_profit {target} ATR / stop_loss {stop} ATR = {rr:.2f}, "
            f"outside the personal range [{rr_min}, {rr_max}]"
        )

    where = "trade_management.trailing_stop"
    trailing = _mapping(tm, "trailing_stop", "trade_management")
    if _enabled(trailing, where):
        activation = _number(trailing, "activation_atr", where)
        _number(trailing, "distance_atr", where)
        if activation >= target:
            raise TradingError(f"{where}.activation_atr must be below the take-profit distance ({target} ATR)")

    where = "trade_management.break_even"
    be = _mapping(tm, "break_even", "trade_management")
    if _enabled(be, where):
        trigger = _number(be, "trigger_atr", where)
        offset = _number(be, "offset_atr", where, strict_low=False)
        if trigger >= target:
            raise TradingError(f"{where}.trigger_atr must be below the take-profit distance ({target} ATR)")
        if offset >= trigger:
            raise TradingError(f"{where}.offset_atr must be below trigger_atr")

    where = "trade_management.partial_closes"
    partial = _mapping(tm, "partial_closes", "trade_management")
    if _enabled(partial, where):
        levels = partial.get("levels")
        if not isinstance(levels, list) or not levels:
            raise TradingError(f"{where}.levels must list at least one level")
        previous, closed = 0.0, 0.0
        for i, level in enumerate(levels):
            if not isinstance(level, dict):
                raise TradingError(f"{where}.levels[{i}] must be a mapping")
            at = _number(level, "at_rr", f"{where}.levels[{i}]")
            closed += _number(level, "close_pct", f"{where}.levels[{i}]", high=100.0)
            if at <= previous or at >= rr:
                raise TradingError(
                    f"{where}.levels: at_rr must increase and stay below the take-profit reward/risk ({rr:.2f})"
                )
            previous = at
        if closed >= 100:
            raise TradingError(f"{where}: partial closes add up to {closed}%; leave part for take-profit or stop")
    return risk, stop, target, rr


def load_risk_profile(path: str | Path | None = None) -> RiskProfile:
    p = Path(path) if path else default_path()
    if p.is_dir():
        p = p / PROFILE_FILE
    if not p.is_file():
        raise TradingError(f"missing {p}: copy templates/workspace/{PROFILE_FILE} to your workspace and adapt it")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TradingError(f"{p.name}: top level must be a mapping")

    warnings: list[str] = []
    balance_usd = _check_account(_mapping(raw, "account", "profile"), warnings)
    max_risk, rr_min, rr_max = _check_limits(_mapping(raw, "limits", "profile"))
    pause = _check_kill_switch(_mapping(raw, "kill_switch", "profile"))
    volatility = _mapping(raw, "volatility", "profile")
    if volatility.get("measure") not in VOLATILITY_MEASURES:
        raise TradingError(f"volatility.measure must be one of {VOLATILITY_MEASURES}")
    if not isinstance(volatility.get("period"), int) or volatility["period"] < 2:
        raise TradingError("volatility.period must be an integer >= 2")
    if volatility.get("at") != "entry":
        raise TradingError("volatility.at must be 'entry': distances use the ATR known before the trade, fixed for its life")
    risk, stop, target, rr = _check_management(_mapping(raw, "trade_management", "profile"), max_risk, rr_min, rr_max)
    return RiskProfile(p, raw, risk, stop, target, rr, pause, balance_usd, warnings)
