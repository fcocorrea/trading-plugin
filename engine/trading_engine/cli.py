"""Engine CLI. The Claude Code commands (commands/*.md) call these functions through bin/trading-engine."""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from . import __version__, catalog
from .core.errors import TradingError
from .core.timeutil import days_between
from .data import store
from .data.ingest import ingest
from .experiment import load_experiment

IC_MODE_ABBR = {"time_series": "ts", "cross_sectional": "xs"}
PENDING = {
    "compare": "run comparison",
}


def _date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise TradingError(f"invalid date '{s}', use YYYY-MM-DD") from e


def _ts(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M")


def _duration(td: dt.timedelta) -> str:
    minutes = int(td.total_seconds() // 60)
    if minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def cmd_sources(_args) -> int:
    print(f"{'id':<20}{'status':<18}{'trades':<8}{'quotes':<8}{'volume':<10}{'aggr_side':<10}")
    for sid, cfg in catalog.load_sources().items():
        c = cfg.get("capabilities", {})
        print(f"{sid:<20}{str(cfg.get('status', '?')):<18}{str(c.get('trades')):<8}"
              f"{str(c.get('quotes')):<8}{str(c.get('volume_unit')):<10}{str(c.get('aggressor_side')):<10}")
    return 0


def cmd_markets(_args) -> int:
    print(f"{'id':<20}{'status':<18}{'implementation':<18}{'session':<18}")
    for mid, cfg in catalog.load_markets().items():
        print(f"{mid:<20}{str(cfg.get('status')):<18}{str(cfg.get('implementation')):<18}"
              f"{str(cfg.get('session')):<18}")
    return 0


def cmd_assets(_args) -> int:
    assets = catalog.load_assets()
    if not assets:
        print("no assets declared: add them to the workspace catalog/assets.yaml (/trading:add-asset)")
        return 0
    print(f"{'symbol':<12}{'market':<18}{'source':<20}streams")
    for sym, cfg in assets.items():
        print(f"{sym:<12}{str(cfg.get('market')):<18}{str(cfg.get('source')):<20}"
              f"{','.join(cfg.get('streams', []))}")
    return 0


def cmd_fetch(args) -> int:
    summary = ingest(args.symbol, _date(args.start), _date(args.end), stream=args.stream)
    print(f"{args.symbol} {args.stream}: ok={len(summary['ok'])} "
          f"cached={len(summary['cached'])} failed={len(summary['failed'])}")
    for d in summary["ok"]:
        if d["warnings"]:
            print(f"  warning {d['day']}: {'; '.join(d['warnings'])}")
    for f in summary["failed"]:
        print(f"  FAILED {f['day']}: {f['error']}")
    return 1 if summary["failed"] else 0


def cmd_check_data(args) -> int:
    res = catalog.resolve(args.symbol)
    mp = store.manifest_path(res.source_id, res.market_id, args.symbol, args.stream)
    by_day: dict = {}
    for e in store.read_manifest(mp):  # the last entry for a day wins
        if store.day_path(res.source_id, res.market_id, args.symbol, args.stream, _date(e["day"])).exists():
            by_day[e["day"]] = e
    if not by_day:
        print(f"no data for {args.symbol} {args.stream}: the manifest has no day with a Parquet file ({mp})")
        return 0
    days = sorted(by_day)
    rows = sum(e["rows"] for e in by_day.values())
    expected = [d.isoformat() for d in days_between(_date(days[0]), _date(days[-1]))]
    missing = [d for d in expected if d not in by_day]
    print(f"{args.symbol} {args.stream}: {days[0]} .. {days[-1]} | days={len(days)} rows={rows}")
    if missing:
        print(f"missing days: {len(missing)} (first: {missing[:5]})")
    warned = [d for d in days if by_day[d].get("warnings")]
    for d in warned[:10]:
        print(f"  warning {d}: {'; '.join(by_day[d]['warnings'])}")
    return 0


def cmd_check_experiment(args) -> int:
    exp = load_experiment(args.path)
    setup = exp.setup
    universe, decision = setup["universe"], setup["decision"]
    print(f"OK {exp.strategy_id} {setup['setup_version']}: {len(universe['symbols'])} symbols on "
          f"{universe['market']}, cadence {decision['cadence']}, requires {universe['requires']}")
    print(f"data from {exp.data_start:%Y-%m-%d} | sealed holdout [{exp.holdout[0]:%Y-%m-%d}, "
          f"{exp.holdout[1]:%Y-%m-%d}) - do not evaluate until the pipeline is frozen")
    for label in exp.labels:
        tag = " (primary)" if label.primary else ""
        print(f"label {label.name}{tag}: horizon {_duration(label.horizon)}, purge {_duration(label.horizon)}, "
              f"trials {exp.trials_by_label[label.name]}")
        for i, f in enumerate(exp.folds_by_label[label.name], 1):
            print(f"  fold {i}: train [{_ts(f.train_start)}, {_ts(f.train_end)}) "
                  f"val [{_ts(f.val_start)}, {_ts(f.val_end)})")
    fits = sum(exp.trials_by_label[name] * len(folds) for name, folds in exp.folds_by_label.items())
    print(f"search accounting: {exp.total_trials} trials across {len(exp.labels)} labels "
          f"({fits} fits); all of them count for the Deflated Sharpe")
    for w in exp.warnings:
        print(f"warning: {w}")
    return 0


def _fmt(value, spec: str) -> str:
    return "-" if value is None or value != value else format(value, spec)


def cmd_run(args) -> int:
    from .runner import run_experiment  # deferred import: only this command needs the research stack

    report = run_experiment(
        args.path, labels=args.label or None, configs=args.config or None, rerun=args.rerun,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    exp = report.experiment
    print()
    print(f"{exp.strategy_id} {exp.setup['setup_version']} - out-of-sample diagnostics across walk-forward folds")
    for label, dropped in report.dropped_rows.items():
        if dropped:
            print(f"{label}: {dropped} rows dropped for warm-up or missing labels")
    header = f"{'label':<14}{'family':<10}{'config':<22}{'run':<14}{'IC mode':<16}{'mean IC':>9}{'HAC t':>8}{'p':>8}{'IC>0':>7}"
    print(header)
    print("-" * len(header))
    failed = 0
    for t in report.trials:
        if t.status != "ok":
            failed += 1
            print(f"{t.label:<14}{t.model_family:<10}{t.config_name:<22}{t.run_id:<14}FAILED  {t.error}")
            continue
        s = t.signal
        mode = f"{IC_MODE_ABBR[s['mode']]}/{s['period']}"
        reused = " (reused)" if t.reused else ""
        print(f"{t.label:<14}{t.model_family:<10}{t.config_name:<22}{t.run_id:<14}{mode:<16}"
              f"{_fmt(s['mean_ic'], '.4f'):>9}{_fmt(s['t_stat'], '.2f'):>8}{_fmt(s['p_value'], '.3f'):>8}"
              f"{_fmt(s['positive_share'], '.0%'):>7}{reused}")
    print()
    print(f"{len(report.trials)} trials this invocation ({failed} failed); every one is in run_log/ and counts "
          "for the Deflated Sharpe. The sealed holdout was not read.")
    print("IC mode: ts = time series per symbol and period, xs = cross-sectional per timestamp.")
    print("Signal before strategy: a baseline whose HAC interval includes zero does not justify climbing the model ladder.")
    return 1 if failed else 0


def cmd_check_risk(args) -> int:
    from .risk import load_risk_profile

    profile = load_risk_profile(args.path)
    account, limits, tm = profile.account, profile.limits, profile.management
    rsi, streak = limits["rsi_filter"], profile.kill_switch["consecutive_losses"]
    rr = limits["reward_risk"]
    print(f"OK {profile.path}")
    print(f"account: {account['broker']} {account['type']} ({account['platform']}), "
          f"{account['initial_balance']} {account['currency']} = {profile.initial_balance_usd:.2f} USD")
    print(f"risk per trade: {profile.risk_per_trade_pct}% of equity (limit {limits['max_risk_per_trade_pct']}%), "
          f"total open risk <= {limits['max_total_risk_pct']}%, one position <= {limits['max_position_pct']}% of equity")
    print(f"stop-loss {profile.stop_atr} ATR | take-profit {profile.take_profit_atr} ATR | "
          f"reward/risk {profile.reward_risk:.2f} (allowed {rr['min']}-{rr['max']})")
    print(f"ATR({profile.raw['volatility']['period']}) at entry | RSI({rsi['period']}) gate: "
          f"no buys above {rsi['no_buy_above']}, no sells below {rsi['no_sell_below']}")
    print(f"kill switch: pause {streak['pause']} after {streak['losses']} consecutive losing trades; drawdown ladder "
          + ", ".join(f"{lvl['drawdown_pct']}% {lvl['action']}" for lvl in profile.kill_switch["drawdown_ladder"]))
    for name in ("trailing_stop", "break_even", "partial_closes"):
        print(f"{name}: {'enabled' if tm[name]['enabled'] else 'disabled'}")
    for warning in profile.warnings:
        print(f"warning: {warning}")
    return 0


def cmd_record(args) -> int:
    from .data.recorder import run as run_recorder  # deferred import: only this command needs websockets

    run_recorder([s.strip() for s in args.symbols.split(",") if s.strip()])
    return 0


def cmd_pending(args) -> int:
    print(f"pending: '{args.name}' is not implemented yet ({PENDING[args.name]}).", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading-engine", description="Multi-market engine of the trading plugin")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="data sources and capabilities").set_defaults(func=cmd_sources)
    sub.add_parser("markets", help="market profiles").set_defaults(func=cmd_markets)
    sub.add_parser("assets", help="catalog assets").set_defaults(func=cmd_assets)

    f = sub.add_parser("fetch", help="download and validate data (both dates inclusive)")
    f.add_argument("symbol")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--stream", choices=["trades", "quotes"], default="trades")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("check-data", help="coverage and warnings of stored data")
    c.add_argument("symbol")
    c.add_argument("--stream", choices=["trades", "quotes"], default="trades")
    c.set_defaults(func=cmd_check_data)

    e = sub.add_parser("check-experiment", help="validate an experiment and print its fold plan")
    e.add_argument("path", help="experiment folder (or its setup.yaml)")
    e.set_defaults(func=cmd_check_experiment)

    rn = sub.add_parser("run", help="train the model menus walk-forward and report model and signal diagnostics")
    rn.add_argument("path", help="experiment folder")
    rn.add_argument("--label", action="append", help="only this label (repeatable)")
    rn.add_argument("--config", action="append", help="only this model config name (repeatable)")
    rn.add_argument("--rerun", action="store_true", help="train again even if an identical run exists")
    rn.set_defaults(func=cmd_run)

    k = sub.add_parser("check-risk", help="validate the personal risk profile of the workspace")
    k.add_argument("path", nargs="?", help="risk_profile.yaml (default: <workspace>/risk_profile.yaml)")
    k.set_defaults(func=cmd_check_risk)

    r = sub.add_parser("record", help="record live bookTicker")
    r.add_argument("--symbols", required=True, help="comma-separated symbols, without spaces")
    r.set_defaults(func=cmd_record)

    for name in PENDING:
        pp = sub.add_parser(name, help=f"pending: {PENDING[name]}")
        pp.add_argument("path", nargs="?")  # accepts the experiment path even though it is unused for now
        pp.set_defaults(func=cmd_pending, name=name)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except TradingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
