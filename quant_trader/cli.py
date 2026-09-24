"""Command line interface.

    python -m quant_trader run          # live trading loop (scans every 5 minutes)
    python -m quant_trader run --once   # a single scan, then exit
    python -m quant_trader train        # (re)train the model from MT5 history now
    python -m quant_trader status       # account, positions, model and recent trades
    python -m quant_trader download     # save MT5 XAUUSD M5 history to CSV
    python -m quant_trader backtest     # walk-forward backtest (CSV, MT5 or synthetic data)
    python -m quant_trader reset-halt   # clear the drawdown kill switch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from .config import BotConfig, ConfigError, load_config

log = logging.getLogger("quant_trader")

# Exit code for problems a restart cannot fix (bad config, real-account guard).
EXIT_FATAL = 2


def setup_logging(cfg: BotConfig, name: str = "bot") -> None:
    log_dir = cfg.path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _mt5_broker(cfg: BotConfig):
    from .broker.mt5_broker import MT5Broker

    return MT5Broker(cfg.mt5, cfg.symbol)


def cmd_run(cfg: BotConfig, args) -> int:
    from .bot import TradingBot

    if args.dry_run:
        cfg.dry_run = True
    bot = TradingBot(cfg, _mt5_broker(cfg))
    if args.once:
        bot.start()
        try:
            bot.maybe_retrain()
            report = bot.run_cycle()
            print(json.dumps(report, default=str, indent=2))
        finally:
            bot.broker.shutdown()
        return 0
    bot.run_forever()
    return 0


def cmd_train(cfg: BotConfig, args) -> int:
    from .features import WARMUP_BARS, prepare_bars
    from .journal import Journal
    from .learner import SelfLearner

    broker = _mt5_broker(cfg)
    broker.connect()
    try:
        spec = broker.symbol_spec()
        n = cfg.learning.train_bars + WARMUP_BARS + cfg.strategy.horizon_bars + 100
        bars = prepare_bars(broker.rates(n), cfg.symbol.default_spread / spec.point)
        learner = SelfLearner(cfg, Journal(cfg.db_path))
        model = learner.retrain(bars, spec.point, "manual")
        print(model.summary() if model else "No model could be trained (see log).")
    finally:
        broker.shutdown()
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
        broker = _mt5_broker(cfg)
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
    broker = _mt5_broker(cfg)
    broker.connect()
    try:
        bars = broker.rates(args.bars)
    finally:
        broker.shutdown()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(out)
    print(f"Saved {len(bars)} bars ({bars.index[0]} -> {bars.index[-1]}) to {out}")
    return 0


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip("<>"): c for c in df.columns}
    if "time" in cols:
        idx = pd.to_datetime(df[cols["time"]])
    elif "date" in cols and "time" not in cols:
        idx = pd.to_datetime(df[cols["date"]])
    else:
        raise ValueError("CSV needs a 'time' column")
    rename = {cols[k]: k for k in ("open", "high", "low", "close", "tick_volume", "spread") if k in cols}
    if "tickvol" in cols:
        rename[cols["tickvol"]] = "tick_volume"
    out = df.rename(columns=rename)
    out.index = pd.DatetimeIndex(idx, name="time")
    keep = [c for c in ("open", "high", "low", "close", "tick_volume", "spread") if c in out.columns]
    return out[keep]


def cmd_backtest(cfg: BotConfig, args) -> int:
    from .backtest import format_stats, run_backtest
    from .broker.base import default_gold_spec
    from .features import prepare_bars
    from .model import InsufficientDataError

    spec = default_gold_spec()
    if args.csv:
        raw = load_csv(args.csv)
    elif args.synthetic:
        from .synthetic import synthetic_gold_bars

        raw = synthetic_gold_bars(args.bars or 40_000, seed=args.seed)
        print("WARNING: synthetic data - this only demonstrates the pipeline, not real performance.")
    else:
        broker = _mt5_broker(cfg)
        broker.connect()
        try:
            spec = broker.symbol_spec()
            raw = broker.rates(args.bars or 100_000)
        finally:
            broker.shutdown()
    if args.bars:
        raw = raw.iloc[-args.bars:]
    bars = prepare_bars(raw, cfg.symbol.default_spread / spec.point)
    print(f"Backtesting {len(bars)} bars {bars.index[0]} -> {bars.index[-1]} "
          f"(retrain every {args.retrain_days} days)...")
    try:
        res = run_backtest(
            bars, cfg, spec,
            retrain_every_bars=int(args.retrain_days * 276),
            start_equity=args.equity,
            commission_per_lot=args.commission,
        )
    except InsufficientDataError as exc:
        print(
            f"\nNot enough history for a walk-forward backtest: {exc}.\n"
            "In MT5 set Tools > Options > Charts > Max bars in chart to Unlimited, open an XAUUSD M5 chart\n"
            "and hold the Home key until no more history loads, then run the backtest again.",
            file=sys.stderr,
        )
        return 1
    print(format_stats(res.stats))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.round({"entry": 3, "exit": 3, "sl_initial": 3, "tp": 3, "profit": 2, "r_multiple": 3, "p": 3}).to_csv(
        out / "trades.csv", index=False
    )
    res.equity.to_csv(out / "equity.csv")
    pd.DataFrame(res.models).to_csv(out / "models.csv", index=False)
    print(f"\nTrades, equity curve and model history written to {out}/")
    return 0


def cmd_reset_halt(cfg: BotConfig, args) -> int:
    from .journal import Journal
    from .risk import RiskState

    journal = Journal(cfg.db_path)
    state = RiskState.from_dict(journal.get_state("risk_state"))
    state.halted = False
    state.halt_reason = ""
    state.peak_equity = 0.0  # re-anchor the drawdown to current equity
    state.cooldown_until = ""
    journal.set_state("risk_state", state.to_dict())
    print("Kill switch cleared. Drawdown will be measured from current equity.")
    return 0


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

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    setup_logging(cfg, args.cmd)
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
