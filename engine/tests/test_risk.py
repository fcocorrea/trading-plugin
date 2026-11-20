from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from trading_engine.cli import main
from trading_engine.core.errors import TradingError
from trading_engine.risk import load_risk_profile

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "workspace" / "risk_profile.yaml"


@pytest.fixture
def profile_path(isolated_env):
    path = isolated_env / "risk_profile.yaml"
    path.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _edit(path: Path, *keys_and_value) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    *keys, last, value = keys_and_value
    node = data
    for key in keys:
        node = node[key]
    node[last] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_template_matches_personal_rules_and_book(profile_path):
    profile = load_risk_profile()
    limits = profile.limits
    assert limits["max_risk_per_trade_pct"] == 1.0
    assert limits["max_total_risk_pct"] == 3.0
    assert limits["max_position_pct"] == 25.0
    assert (limits["reward_risk"]["min"], limits["reward_risk"]["max"]) == (1.5, 3.0)
    assert (limits["rsi_filter"]["no_buy_above"], limits["rsi_filter"]["no_sell_below"]) == (70, 30)
    assert profile.kill_switch["consecutive_losses"]["losses"] == 3
    assert profile.pause == timedelta(hours=2)
    assert [lvl["drawdown_pct"] for lvl in profile.kill_switch["drawdown_ladder"]] == [5, 10, 15, 20, 30]
    assert (profile.stop_atr, profile.take_profit_atr) == (2.0, 3.0)
    assert profile.reward_risk == pytest.approx(1.5)
    assert profile.initial_balance_usd == pytest.approx(100.0)


def test_unconfirmed_broker_conditions_warn(profile_path):
    profile = load_risk_profile()
    assert profile.account["symbols"] == {}, "the template must not declare any symbol"
    assert any("account.symbols is empty" in w for w in profile.warnings)
    assert any("leverage" in w for w in profile.warnings)


def test_symbol_without_confirmed_specs_warns(profile_path):
    _edit(profile_path, "account", "symbols", {"XYZUSDc": {k: None for k in
          ("contract_size", "min_lot", "lot_step", "max_lot", "swap_long", "swap_short")}})
    assert any("XYZUSDc.contract_size" in w for w in load_risk_profile().warnings)


def test_confirmed_broker_conditions_clear_the_warning(profile_path):
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    data["account"].update(leverage=200, margin_call_level_pct=60, stop_out_level_pct=0)
    spec = {"contract_size": 1, "min_lot": 0.01, "lot_step": 0.01, "max_lot": 100, "swap_long": 0, "swap_short": 0}
    data["account"]["symbols"] = {"XYZUSDc": spec}
    profile_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_risk_profile().warnings == []


@pytest.mark.parametrize("edit, message", [
    (("trade_management", "risk_per_trade_pct", 1.5), "exceeds the personal limit"),
    (("limits", "max_total_risk_pct", 0.5), "at least max_risk_per_trade_pct"),
    (("trade_management", "take_profit", "atr_multiple", 2.5), "outside the personal range"),
    (("trade_management", "take_profit", "atr_multiple", 7.0), "outside the personal range"),
    (("trade_management", "stop_loss", "atr_multiple", 0), "out of range"),
    (("limits", "rsi_filter", "no_sell_below", 80), "lower than no_buy_above"),
    (("kill_switch", "consecutive_losses", "pause", "two hours"), "invalid duration"),
    (("kill_switch", "consecutive_losses", "reinstatement", None), "reinstatement"),
    (("kill_switch", "drawdown_ladder", [{"drawdown_pct": 10, "action": "watch"}]), "stops new positions"),
    (("kill_switch", "drawdown_ladder", [{"drawdown_pct": 10, "action": "stop_new"},
                                         {"drawdown_pct": 5, "action": "unwind"}]), "must increase"),
    (("volatility", "measure", "points"), "volatility.measure"),
    (("volatility", "at", "current"), "volatility.at"),
    (("account", "symbols", {"XYZUSDc": {"min_lot": -1}}), "out of range"),
])
def test_rule_violations_are_rejected(profile_path, edit, message):
    _edit(profile_path, *edit)
    with pytest.raises(TradingError, match=message):
        load_risk_profile(profile_path)


def test_partial_closes_must_leave_a_remainder_below_target(profile_path):
    _edit(profile_path, "trade_management", "partial_closes", {"enabled": True, "levels": [{"at_rr": 1.4, "close_pct": 50}]})
    load_risk_profile(profile_path)
    _edit(profile_path, "trade_management", "partial_closes", {"enabled": True, "levels": [{"at_rr": 1.5, "close_pct": 50}]})
    with pytest.raises(TradingError, match="below the take-profit"):
        load_risk_profile(profile_path)
    _edit(profile_path, "trade_management", "partial_closes", {
        "enabled": True, "levels": [{"at_rr": 0.5, "close_pct": 60}, {"at_rr": 1.0, "close_pct": 40}]})
    with pytest.raises(TradingError, match="leave part"):
        load_risk_profile(profile_path)


def test_break_even_and_trailing_must_trigger_before_target(profile_path):
    _edit(profile_path, "trade_management", "break_even", {"enabled": True, "trigger_atr": 3.5, "offset_atr": 0.1})
    with pytest.raises(TradingError, match="break_even.trigger_atr"):
        load_risk_profile(profile_path)
    _edit(profile_path, "trade_management", "break_even", {"enabled": False})
    _edit(profile_path, "trade_management", "trailing_stop", {"enabled": True, "activation_atr": 3.0, "distance_atr": 1})
    with pytest.raises(TradingError, match="trailing_stop.activation_atr"):
        load_risk_profile(profile_path)


def test_missing_profile_points_to_template(isolated_env):
    with pytest.raises(TradingError, match="templates/workspace/risk_profile.yaml"):
        load_risk_profile()


def test_cli_check_risk(profile_path, capsys):
    assert main(["check-risk"]) == 0
    out = capsys.readouterr().out
    assert "10000 USC = 100.00 USD" in out
    assert "total open risk <= 3.0%" in out
    assert "reward/risk 1.50 (allowed 1.5-3.0)" in out
    assert "pause 2h after 3 consecutive losing trades" in out
    assert "warning: broker conditions not confirmed yet" in out
