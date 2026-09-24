# Quant-trader: self-learning XAUUSD bot for MetaTrader 5

An autonomous trading bot that connects to your locally running **MetaTrader 5**
terminal, scans **XAUUSD (gold) only** on every 5-minute candle close, and opens,
manages and closes trades by itself. It trains its own machine-learning model
from your broker's price history, keeps retraining as the market changes, and
learns from the outcome of every trade it takes.

> **Risk warning.** Trading gold on leverage can lose money quickly, including
> more than you expect during news spikes and gaps. Nothing here is financial
> advice and no profit is guaranteed. A model that worked in the past can stop
> working. Run it on a **demo account** for several weeks first. By default the
> bot **refuses to trade a real-money account** until you explicitly allow it.

---

## How it works

Every 5 minutes (a few seconds after each M5 candle closes) the bot:

1. **Syncs** with MT5: records trades that closed (stop, target, time exit),
   updates equity, daily loss and drawdown tracking.
2. **Manages open trades**: moves the stop to break-even at +1R (optional
   trailing stop), closes trades after 3 hours (the model's horizon), and
   closes before the weekend.
3. **Scores the market**: builds ~50 features from the last 6,000 M5 bars
   (trend, momentum, volatility regime, candle shape, session/time of day,
   daily levels, H1/H4 context). Every feature is causal (no look-ahead) and
   normalised by ATR, so it works at any gold price.
4. **Asks the model** for two probabilities: *P(long hits its target before
   its stop)* and *P(short hits its target before its stop)*.
5. **Filters**: trading session, spread, abnormal (news) candles, and risk
   limits (max positions, trades per day, daily loss, cooldown after
   losses, drawdown kill switch).
6. **Executes**: a market order with the stop at 1.5×ATR and the target at
   2.25×ATR (1.5R), sized so the stop loses a fixed % of equity (default
   0.5%) and capped by free margin.

### The self-learning loop

| Mechanism | What it does |
|---|---|
| **Rolling retraining** | Every 24 h the bot downloads the last ~40,000 M5 bars and retrains, so it adapts to the current gold regime. Recent data is weighted more heavily. |
| **Honest validation** | Each model is trained on older data and tested on newer data it has never seen, with a purge gap so labels can't leak. The confidence threshold is chosen on that unseen data. |
| **Quality gates** | A model only trades if its out-of-sample results pass: enough trades, positive expectancy, profit factor > 1.1, t-stat ≥ 2, and a model AUC above chance. **If no edge is found, the bot stays flat** instead of gambling. |
| **Champion / challenger** | A new model replaces the running one only if it is at least as good on data *neither* model was trained on. |
| **Learning from its own trades** | Every real trade (real fill, spread, slippage and management) replaces the simulated label for that bar and counts 3× in the next training run. |
| **Self-suspension** | If the running model starts losing on new data and no better model can be found, trading is suspended until one is. |
| **Adaptive risk** | After a losing streak the bot halves its risk and demands higher confidence. A string of losses also triggers an early retrain. |

Training labels come from a *triple-barrier* simulation that uses the exact
stop, target, holding time, spread and slippage the live bot uses. The
backtester reuses the same filters, risk rules and trade management code as
the live bot, so the backtest measures what the live bot actually does.

---

## Requirements

- **Windows** (the official `MetaTrader5` Python package is Windows-only; a
  Windows VPS is ideal for 24/5 running)
- **MetaTrader 5** terminal installed and logged in to your broker
- **Python 3.10 – 3.14, 64-bit**, from python.org

## Setup

1. **Prepare MT5**
   - Log in to your (demo) account.
   - Enable **Algo Trading** (toolbar button must be green).
   - *Tools → Options → Charts → Max bars in chart*: set to **Unlimited** or
     at least 100000, so enough history is available for training.
   - Open an XAUUSD M5 chart once and scroll back so the terminal
     downloads history.
2. **Install the bot's Python packages** (once). Open a Command Prompt *in the
   Quant-trader folder* (type `cmd` in the Explorer address bar) and run:
   ```bat
   py -m pip install -r requirements.txt
   copy config.example.yaml config.yaml
   ```
   `py` is the Python launcher that python.org installs. Use the same command
   (`py` or `python`) to install and to run the bot, so both use the same Python.
   Alternatively, double-click `setup_windows.bat` to install into an isolated
   `.venv` folder (then `run_bot.bat` uses it automatically).
3. **Configure** `config.yaml`. The defaults are sensible; the main settings
   are `risk.risk_per_trade_pct` and the session hours. Leave the `mt5`
   login fields empty to use the account already logged in in the terminal,
   or set the `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` environment
   variables instead of storing the password in the file.

## Usage

```bat
:: 1. See how the full self-learning system would have traded your broker's data
py -m quant_trader backtest

:: 2. Train now (otherwise the bot trains automatically on first start)
py -m quant_trader train

:: 3. Watch it think without sending orders
py -m quant_trader run --dry-run

:: 4. Trade autonomously (demo first!). Or double-click run_bot.bat,
::    which restarts the bot automatically if it ever crashes.
py -m quant_trader run

:: Other commands
py -m quant_trader run --once       & rem one scan, then exit
py -m quant_trader status           & rem model, risk state, trades, open positions
py -m quant_trader download --bars 100000 --out data\xauusd_m5.csv
py -m quant_trader backtest --csv data\xauusd_m5.csv --retrain-days 5 --commission 7
py -m quant_trader backtest --synthetic   & rem pipeline demo, no MT5 needed
py -m quant_trader reset-halt       & rem clear the drawdown kill switch
```

Stop the bot with **Ctrl+C**. Open trades keep their stop-loss and
take-profit on the broker's server, so they stay protected while the bot is
off. On restart the bot picks them up again.

### What to expect on first start

The bot downloads history, trains, and prints something like:

```
Challenger: model 20260924-140512 [TRADEABLE] thr long=0.52 short=0.55 | validation: 64 trades, exp=+0.182R, PF=1.41 ...
```

If it prints **NOT TRADEABLE** with reasons ("no side shows a positive
out-of-sample edge", "t-stat below 2.00"...), the learner did not find an
edge it trusts in recent data. It will **not trade** and will retry every
6 hours. This is deliberate. You can loosen the gates in the `learning:`
section, but that mostly means trading noise.

### Going live with real money

Only after a demo period you are happy with: set `mt5.allow_real_account: true`
and keep `risk.risk_per_trade_pct` small.

---

## Safety features

- Trades **XAUUSD only**: resolves your broker's gold symbol (`XAUUSD`,
  `XAUUSDm`, `XAUUSD.a`, `GOLD`...) and refuses anything else.
- Refuses **real accounts** unless `allow_real_account: true`.
- Every order carries a **server-side stop-loss and take-profit**.
- **Fixed-fractional sizing** plus a free-margin cap. If even the broker's
  minimum lot would exceed your risk budget, the trade is skipped.
- **Daily loss limit** (default 2%), **max trades per day**, **max open
  positions**, **cooldown** after 3 consecutive losses.
- **Kill switch** at 10% drawdown from the equity peak (clear it with `reset-halt`).
- **No trading** outside session hours, around the daily rollover, late
  Friday, on wide spreads, or right after abnormal news candles.
- Unknown config keys are rejected, so a typo can't silently disable a limit.
- Only manages positions carrying its own **magic number**; your manual
  trades are left alone.

## Files it creates

| Path | Contents |
|---|---|
| `logs/bot.log` | Every scan, signal, order and training run (rotated) |
| `data/journal.sqlite` | Trades (with the features at entry), signals, equity, model history, risk state |
| `models/champion.joblib` | The model currently trading, plus recent history models |
| `backtest_results/` | `trades.csv`, `equity.csv`, `models.csv` from the last backtest |

## Project layout

```
quant_trader/
  bot.py          live loop: 5-minute scans, entries, management, sync
  learner.py      self-learning: retraining schedule, champion/challenger, live feedback
  model.py        gradient-boosted long/short models, threshold selection, quality gates
  features.py     causal, scale-free feature engineering (M5 + H1/H4)
  labeling.py     triple-barrier labels matching the live stop/target/horizon
  policy.py       shared trading rules (decisions, filters, stops, trade management)
  risk.py         position sizing, daily loss, drawdown kill switch, adaptive risk
  backtest.py     walk-forward backtest of the whole system
  journal.py      SQLite journal
  broker/         MT5 connector + simulator
  cli.py          command line interface
tests/            unit and integration tests (run anywhere, no MT5 needed)
```

## Troubleshooting

| Message | Fix |
|---|---|
| `No module named 'yaml'` (or numpy, pandas, sklearn) / `Missing Python packages` | The packages aren't installed for the Python you ran. In the Quant-trader folder run `py -m pip install -r requirements.txt` (the bot prints the exact command for your Python). |
| `MetaTrader5 package is not installed` / pip can't find `MetaTrader5` | MetaTrader5 exists only for **64-bit Windows Python 3.10 – 3.14**. Install that from python.org, then repeat the pip command. |
| `Could not connect to MetaTrader 5` | Open the MT5 terminal and log in. With several terminals installed, set `mt5.terminal_path` in `config.yaml`. |
| `Not enough history for a walk-forward backtest` | In MT5 set *Tools → Options → Charts → Max bars in chart* to Unlimited, open an XAUUSD M5 chart and hold **Home** until no more history loads. |
| `Algo Trading is disabled` | Click the **Algo Trading** button in the MT5 toolbar (it turns green). |
| `NOT TRADEABLE` after training | Not an error: no statistically reliable edge was found in recent data, so the bot stays flat and retries later. |

## Development

```bash
py -m pip install -r requirements-dev.txt
py -m pytest
```

The tests run on Linux/macOS too: a simulated broker replays synthetic gold
bars through the real bot loop, and the MT5 connector is tested against a
fake `MetaTrader5` module.

## Notes and limitations

- Hours in the config are **broker server time** (usually UTC+2 in winter and
  UTC+3 in summer, aligned to the New York close).
- The scan runs on your PC clock; keep Windows time synced.
- The bot must keep running to manage trades and learn. Disable sleep, or use a VPS.
- Backtests include spread and a slippage allowance, and optionally
  commission (`--commission` per lot, round turn), but not swap. Live
  results will differ from backtests.
