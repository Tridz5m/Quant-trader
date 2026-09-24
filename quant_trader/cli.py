"""Command line interface.

    python -m quant_trader run          # live trading loop (scans every 5 minutes)
    python -m quant_trader run --once   # a single scan, then exit
    python -m quant_trader train        # (re)train the model from MT5 history now
    python -m quant_trader status       # account, positions, model and recent trades
    python -m quant_trader download     # save MT5 XAUUSD M5 history to CSV
    python -m quant_trader backtest     # walk-forward backtest (CSV, MT5 or synthetic data)
    python -m quant_trader reset-halt   # clear the drawdown kill switch
    python -m quant_trader app          # open the desktop app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import services
from .config import BotConfig, ConfigError, load_config
from .logsetup import setup_logging

log = logging.getLogger("quant_trader")

# Exit code for problems a restart cannot fix (bad config, real-account guard).
EXIT_FATAL = 2


def cmd_run(cfg: BotConfig, args) -> int:
    from .bot import TradingBot

    if args.dry_run:
        cfg.dry_run = True
    bot = TradingBot(cfg, services.mt5_broker(cfg))
    if args.once:
        print(json.dumps(bot.run_once(), default=str, indent=2))
        return 0
    bot.run_forever()
    return 0


def cmd_train(cfg: BotConfig, args) -> int:
    model = services.train_from_mt5(cfg)
    print(model.summary() if model else "No model could be trained (see log).")
    return 0


def cmd_status(cfg: BotConfig, args) -> int:
    from .journal import Journal
    from .learner import SelfLearner

    journal = Journal(cfg.db_path)
    learner = SelfLearner(cfg, journal)
    print("Model:", json.dumps(learner.status(), default=str, indent=2))
    print("Risk state:", json.dumps(journal.get_state("risk_state", {}), indent=2))
    closed = journal.closed_trades(20)
    if closed:
        df = pd.DataFrame(closed)[["ticket", "direction", "volume", "entry_time", "exit_time", "profit", "r_multiple", "close_reason"]]
        print("\nLast closed trades:\n", df.to_string(index=False))
        rs = [t["r_multiple"] for t in closed if t["r_multiple"] is not None]
        if rs:
            print(f"\nLast {len(rs)} trades: expectancy {sum(rs) / len(rs):+.2f}R, win rate {sum(r > 0 for r in rs) / len(rs):.0%}")
    if not args.offline:
        broker = services.mt5_broker(cfg)
        broker.connect()
        try:
            acc = broker.account()
            print(f"\nAccount {acc.login} {'REAL' if acc.is_real else 'DEMO'}: balance {acc.balance:.2f} equity {acc.equity:.2f} {acc.currency}")
            for p in broker.positions():
                side = "BUY" if p.direction > 0 else "SELL"
                print(f"  #{p.ticket} {side} {p.volume} @ {p.price_open} SL {p.sl} TP {p.tp} P/L {p.profit:.2f}")
        finally:
            broker.shutdown()
    return 0


def cmd_download(cfg: BotConfig, args) -> int:
    out, n = services.export_history(cfg, Path(args.out), args.bars)
    print(f"Saved {n} bars to {out}")
    return 0


def cmd_backtest(cfg: BotConfig, args) -> int:
    from .backtest import format_stats
    from .model import InsufficientDataError

    source = "csv" if args.csv else "synthetic" if args.synthetic else "mt5"
    try:
        bars, spec, margin_rate = services.load_backtest_bars(cfg, source, args.csv, args.bars, args.seed)
    except (OSError, ValueError) as exc:
        print(f"Could not load the data: {exc}", file=sys.stderr)
        return 1
    if source == "synthetic":
        print("WARNING: synthetic data - this only demonstrates the pipeline, not real performance.")
    print(f"Backtesting {len(bars)} bars {bars.index[0]} -> {bars.index[-1]} "
          f"(retrain every {args.retrain_days} days)...")
    try:
        res = services.backtest(cfg, bars, spec, args.retrain_days, args.equity, args.commission, margin_rate)
    except InsufficientDataError as exc:
        print(f"\nNot enough history for a walk-forward backtest: {exc}.\n{services.MORE_HISTORY_HELP}", file=sys.stderr)
        return 1
    print(format_stats(res.stats))
    out = services.save_backtest(res, Path(args.out), bars if source == "mt5" else None)
    print(f"\nTrades, equity curve and model history written to {out}/")
    return 0


def cmd_reset_halt(cfg: BotConfig, args) -> int:
    services.reset_halt(cfg)
    print("Kill switch cleared. Drawdown will be measured from current equity.")
    return 0


def cmd_app(cfg: BotConfig, args) -> int:
    from .app import main as app_main

    return app_main([])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant_trader", description="Self-learning XAUUSD bot for MetaTrader 5")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML (default: config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the live trading loop")
    p.add_argument("--once", action="store_true", help="run a single scan and exit")
    p.add_argument("--dry-run", action="store_true", help="log signals without sending orders")

    sub.add_parser("train", help="retrain the model from MT5 history now")

    p = sub.add_parser("status", help="show model, risk state, trades and positions")
    p.add_argument("--offline", action="store_true", help="do not connect to MT5")

    p = sub.add_parser("download", help="save MT5 M5 history to CSV")
    p.add_argument("--bars", type=int, default=100_000)
    p.add_argument("--out", default="data/xauusd_m5.csv")

    p = sub.add_parser("backtest", help="walk-forward backtest of the self-learning system")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="CSV with time,open,high,low,close[,tick_volume,spread]")
    src.add_argument("--synthetic", action="store_true", help="use synthetic data (pipeline demo)")
    p.add_argument("--bars", type=int, default=None, help="use only the last N bars")
    p.add_argument("--retrain-days", type=float, default=5.0)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--commission", type=float, default=0.0, help="round-turn commission per lot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="backtest_results")

    sub.add_parser("reset-halt", help="clear the max-drawdown kill switch")
    sub.add_parser("app", help="open the desktop app")

    args = parser.parse_args(argv)
    if args.cmd == "app":
        return cmd_app(None, args)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    setup_logging(cfg, "bot" if args.cmd == "run" else args.cmd)
    handlers = {
        "run": cmd_run,
        "train": cmd_train,
        "status": cmd_status,
        "download": cmd_download,
        "backtest": cmd_backtest,
        "reset-halt": cmd_reset_halt,
    }
    from .bot import SafetyError
    from .broker.base import BrokerError

    try:
        return handlers[args.cmd](cfg, args)
    except SafetyError as exc:
        log.error("SAFETY STOP: %s", exc)
        return EXIT_FATAL
    except BrokerError as exc:
        log.error("MT5 error: %s", exc)
        return 1
