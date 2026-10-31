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
pytest                                    # 98 tests, ~25 s

# No network needed — synthetic data, proves the plumbing:
python -m tradebot.run.backtest --synthetic trending --strategy momentum

# Real data (needs exchange API access from this host):
python -m tradebot.run.fetch              # pins a content-addressed snapshot
python -m tradebot.run.backtest --strategy momentum --n-trials 12
python -m tradebot.run.walkforward --strategy momentum

# Paper trading (simulated fills, live quotes, no API keys):
python -m tradebot.run.paper --replay --synthetic trending --max-bars 500
python -m tradebot.run.paper --strategy momentum        # long-running
python -m tradebot.run.paper --status                   # how is it doing
python -m tradebot.run.paper --watchdog                 # is it still alive
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

## Paper trading

Paper mode needs no credentials: it reads public market data and simulates
fills locally with the **same** strategy, risk gate, accounting and cost model
as a backtest. That parity is the point — `tests/test_paper.py` asserts that
replaying a snapshot through the paper loop reproduces the backtest's equity
curve bar for bar, so a later divergence on live data is information about the
market rather than about which code path ran.

What it does that a backtest cannot:

| Concern | How it is handled |
|---|---|
| Restarts | Every decision, order and fill is journalled (`state/<run>/journal.jsonl`) before it matters. On startup the book is rebuilt by replaying fills, and bars already processed are skipped rather than re-traded. |
| Double fills | `client_order_id` is derived from the bar, so a replayed bar regenerates the same id and it is rejected. The seen-set is seeded from the journal, so this survives a restart. |
| Unknown state | Startup compares the journal-derived book against the checkpoint and **refuses to start** on a mismatch. |
| A breach | The gate flattens the book, writes a sticky `HALTED` sentinel, and exits 2. It will not restart until a human runs `--clear-halt`. The drawdown high-water mark is replayed from the journal, so a crash loop cannot reset its own stops. |
| Silent death | A heartbeat is written each bar; `--watchdog` reads it from outside the process and exits non-zero on a stale heartbeat or a halt. It is wired to the Compose healthcheck. |

Exit codes, because something other than a human reads them: `0` clean stop,
`1` refused to start, `2` **halted by the risk gate**, `3` watchdog found a
dead or halted run.

Two honest caveats. Paper fills are *modelled*, not executed — the impact term
is a prior, and calibrating it is what this phase is for. And in paper mode
there is no venue to ask for positions, so the startup check verifies our own
journal against our own checkpoint; a live adapter would have to pass real
venue positions to `reconcile()` instead.

Run `--replay` before trusting a deployment: it drives the whole daemon,
journal and reconciliation included, from pinned data at CPU speed.

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
                     (~0.11 ms/bar; output pinned bit-for-bit by
                      tests/test_engine_golden.py)
  validation/
    metrics.py       Sharpe + standard error, PSR, deflated Sharpe
    walkforward.py   rolling windows + parameter-stability reporting
    pbo.py           probability of backtest overfitting (CSCV)
  execution/
    sim.py           backtest fills
    paper.py         simulated fills vs live quotes; durable idempotency
    live.py          raises on import, deliberately
  state.py           append-only journal, book checkpoint, heartbeat, halts
  data/quotes.py     live + replay quote sources; drops the unclosed bar
  engine/paper.py    the paper loop: same bar ordering as the backtest
  run/               CLIs: fetch, backtest, walkforward, paper
```

## Deployment (Hetzner)

```bash
docker compose run --rm fetch         # pin a snapshot
docker compose run --rm backtest      # measure
docker compose run --rm paper-replay  # dry-run the daemon against pinned data
docker compose --profile paper up -d  # long-running paper trader
```

The `paper` service restarts `unless-stopped` and its healthcheck is the
watchdog, so a stale heartbeat or a halt shows up as an unhealthy container.
Compose will not wake anyone by itself: wiring `--watchdog` to something that
reaches a human is the last open item before live trading could be considered
at all (DESIGN.md §7).

## Status

Working: data pinning and validation, cost model, risk gate, backtest engine,
walk-forward, PBO, deflated Sharpe, **and the paper-trading daemon** —
journalled state, startup reconciliation, durable idempotency, sticky halts,
heartbeat and watchdog. 98 tests.

**Nothing has been measured on real data yet.** Exchange APIs are not
reachable from the sandbox this was built in, so every number produced so far
comes from synthetic data and proves plumbing, not profitability. The next
piece of work is on the host: `run.fetch` to pin a real snapshot, then
walk-forward, then paper trading long enough to span a regime change.

See [DESIGN.md §11](DESIGN.md#11-known-limitations) for the full honest list
of limitations.
