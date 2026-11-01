# tradebot

A **crypto spot strategy research harness**: backtesting, walk-forward
validation, and paper trading, built so that a result you get out of it is one
you can believe.

**Paper trading only.** There is no live order path — `execution/live.py`
raises on import, and there are no live credentials in the config. See
[DESIGN.md §7](DESIGN.md#7-live-trading-is-absent-deliberately) for the gate
that would have to be cleared to change that.

Read [**DESIGN.md**](DESIGN.md) before changing anything. It is the rulebook:
the no-lookahead contract, the cost model, what the risk limits are actually
for, and the validation discipline.

---

## What problem this solves

The hard part of automated trading is not the infrastructure. It is knowing
whether you have an edge. The evidence on that is unkind — Barber et al. found
**under 1%** of Taiwanese day traders were predictably profitable after fees;
ESMA puts retail CFD accounts at **74–89%** losing — and essentially all of
the usual ways a backtest lies are avoidable:

- looking ahead, even by one bar
- ignoring or understating costs
- tuning on the test set
- reporting the best of fifty variants as if you had tried one
- measuring a Sharpe ratio over a sample far too short to measure anything

This harness is built to make each of those either impossible or loudly
visible. The success criterion is not "it made money" — it is **"it told me
the truth quickly."**

## Quick start

```bash
pip install -e ".[dev]"
pytest                                    # 45 tests, ~2 min

# No network needed — synthetic data, proves the plumbing:
python -m tradebot.run.backtest --synthetic trending --strategy momentum

# Real data (needs exchange API access from this host):
python -m tradebot.run.fetch              # pins a content-addressed snapshot
python -m tradebot.run.backtest --strategy momentum --n-trials 12
python -m tradebot.run.walkforward --strategy momentum

# Paper trading (public live quotes, simulated fills, no keys):
python -m tradebot.run.paper --strategy momentum
# ... or rehearse the daemon end to end with no network:
python -m tradebot.run.paper --replay-synthetic trending

# Watch any paper run in a browser (equity curve, trades, heartbeat):
python -m tradebot.run.dashboard        # http://127.0.0.1:8060
```

`--n-trials` is how many variants you have tried **in total** for this idea.
It feeds the deflated Sharpe. Understating it only fools you — keep the count
in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Reading the output

The report leads with anything that invalidates it, then warns in plain
language. Four things to look at before the headline return:

| Field | What it means |
|---|---|
| `sharpe_t_stat` | Below ~2, the Sharpe is **not distinguishable from zero**. 62 days of hourly data gives a standard error of 2.4. |
| `clamp_rate` | Fraction of decisions the risk gate overrode. High means the result measures the gate, not the strategy. |
| `truncated` / `halted` | A hard stop fired and everything below describes a **stub** of the sample. |
| `deflated_sharpe` | Credibility after accounting for how many variants you tried. |

Plus `excess cagr vs hold`: over most crypto windows, a strategy that does not
beat buy-and-hold after costs has produced nothing but turnover.

## Layout

```
src/tradebot/
  types.py           MarketView — makes lookahead structurally impossible
  costs.py           fees + half-spread + sqrt-law impact (never optional)
  risk.py            the gate: only ever reduces exposure or halts
  config.py          validated on load
  data/
    store.py         content-addressed, hash-verified OHLCV snapshots
    fetch.py         ccxt downloader (public endpoints, no keys)
    synthetic.py     deterministic generators, incl. the random walk used
                     to detect lookahead bugs
  strategy/          pure functions of history; no state, no I/O
  engine/
    portfolio.py     spot accounting: no shorting, no negative cash
    backtest.py      decide at bar i, fill at i+1 open, mark at i+1 close
  validation/
    metrics.py       Sharpe + standard error, PSR, deflated Sharpe
    walkforward.py   rolling windows + parameter-stability reporting
    pbo.py           probability of backtest overfitting (CSCV)
  execution/
    sim.py           backtest fills
    live.py          raises on import, deliberately
  paper/
    daemon.py        the long-running loop; same decision path as the backtest
    feed.py          closed bars only: live ccxt polling, or deterministic replay
    journal.py       append-only journal, snapshot, heartbeat, HALTED marker
    alerts.py        stderr + optional webhook; halts must reach a human
    report.py        read-only fold of a run's journal, for the dashboard
    dashboard.html   the dashboard page: stdlib-served, no CDN, light/dark
  run/               CLIs: fetch, backtest, walkforward, paper, dashboard
```

## Deployment (Hetzner)

```bash
docker compose run --rm fetch       # pin a snapshot
docker compose run --rm backtest    # measure
docker compose up -d paper          # long-running paper trader
```

The paper service restarts on crashes and carries a healthcheck wired to the
daemon's heartbeat file. A run halted by the risk gate stays halted across
restarts until a human runs `--clear-halt` — Docker cannot quietly resume it.

## Status

Working: data pinning and validation, cost model, risk gate, backtest engine,
walk-forward, PBO, deflated Sharpe, and the paper-trading daemon (journaled
state, idempotent restarts, startup reconciliation, persistent halts,
heartbeat). 55 tests, including a proof that the daemon reproduces
`run_backtest` fill-for-fill on replayed data.

**Not yet done: a paper run on real data.** The daemon exists; what it has
not had yet is weeks of unattended runtime against live quotes, which is what
actually validates it. See [DESIGN.md §10](DESIGN.md#10-known-limitations)
for the full honest list of limitations.
