# Handoff — 2026-11-01

A short note so a new session can pick this up without the conversation that
produced it. **Read [DESIGN.md](DESIGN.md) first** — it is the rulebook, and it
explains not just what the rules are but what went wrong when each was missing.

This file is disposable. Delete it once it is stale; the repo documents itself.

---

## Where the work is

Branch **`claude/modest-lamport-aecdvp`**, three commits ahead of `main`, not
merged and with no pull request open:

```
4efc14d  Fix: `docker compose run --rm fetch` failed on a fresh clone
bd1a32b  Make the engine ~10x faster, with output pinned bit-for-bit
2efee27  Paper-trading daemon: journalled state, sticky halts, backtest parity
```

98 tests, ~25s. Everything is committed and pushed.

## The one thing that matters

**Nothing has ever been measured on real market data.** Every number this
harness has produced came from synthetic generators, which prove the plumbing
runs and prove nothing whatever about profitability. Momentum scoring 14x on
trend-generated data is a tautology, not a finding.

The reason is environmental, not technical: this was built in a cloud sandbox
whose network policy blocks every crypto exchange (twelve were tested — all
refused at the proxy, while ordinary hosts resolve fine). Pinning a real
snapshot has to happen on a machine with exchange access.

So the next step is not a code change. It is pointing the instrument at
something real.

## Next steps, in order

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # expect 98 passed

python -m tradebot.run.fetch                # needs exchange access
```

Then, and this order is deliberate:

1. **Read the fetch warnings before using the data.** Expect ~51,000 hourly
   bars for the configured window. `validation_problems` listing irregular bar
   intervals means missing candles, which silently shifts every lookback
   window in every calculation downstream.
2. **Commit the manifest** (`data/snapshots/*.manifest.json`). The parquet is
   gitignored on purpose — large and rebuildable — but the manifest pins the
   content hash, so a later re-fetch that does not match tells you the
   exchange revised its history. That is a real phenomenon and the whole
   reason snapshots are content-addressed.
3. **One backtest against the unconstrained buy-and-hold benchmark.** Pass an
   honest `--n-trials`. Over most crypto windows, a strategy that does not
   beat holding after costs has produced nothing but turnover.
4. **Walk-forward, and log it in [RESEARCH_LOG.md](RESEARCH_LOG.md)** —
   including variants you abandon. The deflated Sharpe is only meaningful
   against a complete trial count, and understating it fools nobody but you.
5. **Only then consider paper trading**, and not on a laptop: it is meant to
   run unattended for one to three months spanning a regime change, and every
   hour a machine is asleep is an hour of missed decisions that makes the
   paper record incomparable to the backtest it is being checked against.
   That is what the Hetzner box is for.

Sanity check the whole daemon first, offline and at CPU speed:

```bash
python -m tradebot.run.paper --replay --strategy momentum --max-bars 2000
```

## What to expect

Most likely the walk-forward says there is no edge. That is the instrument
working: it exists to reach that verdict in an afternoon rather than after six
months of paying fees to find out. A disappointing first result is not a
reason to start tuning — see DESIGN.md §9 on what tuning after a disappointing
out-of-sample number actually costs you.

## Two conventions worth not breaking

**The engine's output is pinned bit-for-bit.** `tests/test_engine_golden.py`
hashes the exact float repr of every equity point, fill and violation across
seven cases. If a change makes those fail, it changed behaviour — which is
fine if intended and a silent disaster if not, because every result recorded
before it stops being comparable and nothing in a returns series announces
that it moved. Regenerating the digests is a deliberate act with its own
commit message.

**Paper and backtest must stay the same computation.** `tests/test_paper.py`
asserts that replaying a panel through the paper loop reproduces the
backtest's equity curve bar for bar. That is what makes a live divergence
informative rather than ambiguous.

## Still open

See DESIGN.md §11 for the full list. The two that matter:

- **Alerting does not reach a human.** `--watchdog` exits non-zero and the
  container healthcheck goes unhealthy, but nothing pages anybody. This is
  item 5 of the §7 live-trading gate and is still unbuilt.
- **Paper reconciliation has no counterparty.** It verifies the journal
  against our own checkpoint, which catches corruption but is not a venue
  check. A live adapter must pass real venue positions into `reconcile()`.

A walk-forward is embarrassingly parallel across fold × parameter combination
and currently runs on one core. Worth doing when a grid gets large enough to
be annoying, and not before — it is about a minute as it stands.
