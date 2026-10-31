# Design and discipline

This document is the rulebook. The code enforces most of it; where it cannot,
this is the written record of what we decided and why.

---

## 1. What this is, and what it is not

It is a **measurement instrument**: a harness for deciding, cheaply and
honestly, whether a trading idea has an edge that survives costs.

It is **not** a money machine, and it is not designed as one. That framing is
a response to the evidence, not modesty:

| Study | Population | Finding |
|---|---|---|
| Barber et al. (Taiwan, every TSE trade 1992–2006) | all day traders | **under 1%** predictably profitable after fees |
| Chague et al. (2020, Brazil index futures) | persisted >300 days | **97%** lost money |
| ESMA (national regulator analyses, EU) | retail CFD accounts | **74–89%** lose money |

The binding constraint on a retail trading system is not infrastructure — the
infrastructure is a solved problem, and most of it is in this repo. The
binding constraint is **edge**, and no amount of good engineering manufactures
it. What good engineering does is stop you believing you have edge when you
do not. That is the entire purpose of this codebase.

So the measure of success for this harness is not "it made money." It is
**"it told me the truth quickly."** A harness that kills a bad idea in a day
has done its job.

---

## 2. Shape of the system

```
market data ──► strategy (PURE) ──► risk gate ──► execution adapter ──► venue
   (pinned          │                   │               │
    snapshot)       │                   │          sim │ paper │ live(absent)
                    │                   │
                    └───────────────► append-only state: decisions, orders, fills
                                            │
                                   monitoring / alerts / kill switch
```

One rule governs the whole picture: **everything above the execution adapter
is identical in every mode.** Same strategy object, same risk gate, same
accounting, whether backtesting or paper trading. That parity is what makes a
divergence between backtest and paper *informative* — it means something real
changed, rather than that you took a different code path.

---

## 3. No lookahead, by construction

The contract, enforced in `engine/backtest.py`:

> At bar `i`, the strategy sees history **up to and including** bar `i`'s
> close. Resulting orders fill at bar **`i+1`'s open**, with costs. Equity is
> marked at bar `i+1`'s close.

This is not conservatism. At the moment a bar closes, its close is the newest
thing you know, and the soonest you can act is the next bar.

The guarantee is **structural, not procedural**. A strategy receives a
`MarketView`, which holds a slice ending at bar `i`. Future bars are *absent
from the object* — a strategy cannot cheat without reaching outside its own
arguments. Reviewing for lookahead is unreliable; making it unrepresentable is
not.

Three tests defend this (`tests/test_no_lookahead.py`):

1. Every fill lands on a strictly later bar than the decision that caused it.
2. A deliberately cheating control (`FutureOracle`, which holds the full panel
   and reads the next bar) scores Sharpe ≈ **35** and turns 10,000 into
   **40,000,000**, against ≈ 1 and ≈ 11,000 for the honest path. If those
   margins ever narrow, the engine has started leaking.
3. On a driftless random walk — unpredictable by construction — no strategy
   earns a meaningful risk-adjusted return after costs.

### Strategies must be pure

No state across calls, no I/O, no wall clock. Remembered state means replaying
a bar after a restart gives a different answer, at which point the backtest no
longer describes the live system and crash recovery silently changes
behaviour. `CrossSectionalMomentum` recomputes its held position from history
rather than remembering it, deliberately, and the waste is worth it.

---

## 4. Costs are the hurdle, and they are not optional

There is no "costs off" mode. Three components, all in bps of traded notional:

- **fee** — exchange commission (~10 bps Binance taker, ~26 bps Kraken)
- **half-spread** — you buy at the ask and sell at the bid
- **impact** — square-root law (Almgren): impact scales with √(participation
  in the bar's volume)

The number that matters, from the defaults in `config/default.yaml`:

> **A round trip at 1% participation costs ~54 bps.**

Any signal whose average edge per trade is below that loses money with
perfect execution. Most simple technical signals are well below it. Before
tuning a strategy, check whether its per-trade edge could clear 54 bps even in
principle.

Two honesty rules:
- Lowering the cost assumptions to improve a backtest has improved nothing.
- Unknown volume assumes **worst-case** impact, never free execution.

Use `python -m tradebot.run.backtest --cost-attribution` to run the same
strategy frictionlessly and see the gap. Caveat: that run also drops
`min_notional`, so it takes a slightly different trade path — it is a
diagnostic, not an exact decomposition.

---

## 5. Risk limits are a circuit breaker, not risk management

`risk.py` sits between strategy and execution in every mode. It **only ever
reduces exposure or halts**, never raises size or invents a trade, which makes
it safe to apply blindly.

**The calibration lesson, learned the hard way.** The first version defaulted
to `max_daily_loss_pct = 0.05`. On 1h crypto data, a buy-and-hold benchmark
halted **16 hours in**, and every backtest silently truncated to under a day.
The reason is structural: a long-only, fully-invested crypto book *tracks the
market*, and crypto routinely moves 5–10% in a day. A 5% daily stop therefore
fires on **market** moves, not strategy errors, and exits permanently at the
worst possible moment.

So treat these stops as a guard against **bugs and runaway behaviour** — a
strategy looping on orders, a sign error, a stale feed. **Position sizing is
what manages market risk.** Defaults are now 15% daily / 40% drawdown, and
crypto buy-and-hold has repeatedly drawn down more than 70%, so even 40% is a
decision to exit most bear markets permanently. Make it deliberately.

`halt_on_breach=False` is the research escape hatch: breaches are still
recorded, but the sample stays intact. **Paper and live runs must leave it
`True`** — a circuit breaker that does not break the circuit is decoration.

Every override is recorded, and `clamp_rate` is reported. A high clamp rate
means **the backtest describes the gate's behaviour, not the strategy's**, and
should not be read as a measurement of the idea. This is not hypothetical: an
early run clamped two different strategies to the same 0.25/0.25/0.25 book and
produced byte-identical results for both.

---

## 6. Idempotency and reconciliation

`client_order_id` is derived deterministically from
`(run_id, decision timestamp, symbol, side)` — never a counter or a clock. A
replayed bar regenerates the **same** id, so the venue rejects the duplicate
instead of doubling the position. A random UUID here is a silent double-fill
waiting for a restart.

On startup, a paper or live run must fetch actual venue positions, compare
against internal state, and **halt on mismatch** rather than trade blind.

How much of that the paper daemon can actually do, stated plainly: the book is
rebuilt by replaying every recorded fill, and compared against the `book.json`
checkpoint. That catches a truncated journal, a hand-edited book, and a crash
between the two writes. It is **not** the real check, because in paper mode
there is no independent counterparty to ask — `PaperExecution.venue_positions()`
returns `None` on purpose rather than handing back our own numbers dressed up
as a venue reply, which would make the check a tautology that passes while
telling you nothing. `reconcile()` reports which comparison it actually ran,
and a live adapter must pass real venue positions into it.

And the failure that actually costs money is not a bad trade. It is the
process **dying silently while holding a position**. A heartbeat and a
dead-man's switch are not optional extras for anything that runs unattended.
The daemon writes `heartbeat.json` every bar; `--watchdog` reads it from
*outside* the process, because nothing inside a dead process can report its
own death. It is wired to the Compose healthcheck. Routing that signal to a
human is still unbuilt, and is item 5 of the live-trading gate below.

---

## 7. Live trading is absent, deliberately

`execution/live.py` raises on import. There are no live credentials in
`.env.example`, because there is no live code path. Enabling it is a reviewed
change, not a config flag.

The gate — all of it, not some of it:

1. A walk-forward out-of-sample Sharpe that survives the **deflated Sharpe**
   adjustment for the number of variants actually tried.
2. **PBO below ~0.3** across the candidate family.
3. One to three months of paper trading **spanning a regime change**, with
   live-versus-backtest tracking error you can explain.
4. A reconciliation loop that halts on any state mismatch.
5. Alerting that reaches a human when the process dies holding a position.
6. A written decision about size that you would be **content to lose
   entirely**.

Then start at 1–2% of intended size and scale only when live tracking matches
paper.

---

## 8. The paper daemon

Same strategy object, same gate, same accounting, same cost model as a
backtest — and the loop body is deliberately the same six steps in the same
order (`engine/paper.py` mirrors `engine/backtest.py`):

    at the close of bar k:  mark equity at bar k's close   <- marks the
                                                              previous
                                                              decision's fill
                            observe equity in the gate
                            check the stops
                            strategy sees history up to and including k
                            gate clamps what it asked for
                            orders sized at bar k's close
    immediately after:      they fill at bar k+1's OPEN, with costs

Parity is asserted, not asserted-to: replay a panel through the paper loop and
the equity curve equals the backtest's on that panel, bar for bar, fills
included (`tests/test_paper.py`). If that test ever fails, every subsequent
paper-versus-backtest comparison is measuring a code difference instead of a
market one, and nobody would be able to tell which.

Four decisions in there that are easy to get wrong:

**The unclosed bar is never served.** `fetch_ohlcv` returns the in-progress
candle last. Its close changes every few seconds, so feeding it to a strategy
makes the same question return different answers, which silently breaks both
reproducibility and parity. It is dropped every poll.

**The fill price is the new bar's open, not the last trade.** Once bar `k+1`
has started, its *open* is already fixed and will never be revised, even
though its close has not happened. Reading that one field is both faithful to
the section 3 contract and reproducible; a ticker price is neither. The ticker
is a fallback for the seconds before the new bar prints.

**Volume at fill time is unknowable, so it is estimated, and the estimate is
labelled.** The backtest uses the actual volume of the bar it fills into. Live,
that bar has not finished, so the daemon uses the bar that just closed as its
estimator and marks the quote `volume_is_estimate`. The fully pessimistic
alternative (assume zero volume, hence maximum impact) is available as
`volume_mode="worst_case"` and is what section 4 requires where there is no
information at all — but here there *is* information, and pretending otherwise
would put a 150 bps impact charge on every paper fill and make the paper run
incomparable to its backtest. **This is the most likely source of
paper-versus-backtest slippage divergence, and measuring it is one of the
things this phase is for.**

**A halt is sticky.** The gate flattens, the daemon writes a `HALTED` sentinel
and exits 2, and the next start refuses. This matters because
`restart: unless-stopped` would otherwise bring the process straight back up
and resume trading — a breaker that trips and immediately re-closes. Clearing
it is an explicit human command. Relatedly, the drawdown high-water mark is
replayed from the journal on startup: rebuild the gate from nothing each time
and a crash-looping process would declare every new, lower equity to be its
peak and could never breach a drawdown limit however far it fell.

---

## 9. Validation discipline

**Sample size first.** The standard error of an annualised Sharpe is
≈ √(bars_per_year / n):

| Sample | SE(annualised Sharpe) |
|---|---|
| 1,500 hourly bars (~62 days) | **2.42** |
| 6,000 bars (~250 days) | 1.21 |
| 3 years hourly | 0.58 |
| 10 years hourly | 0.32 |

At 62 days, a Sharpe of 2.0 is **indistinguishable from zero**. No strategy
work fixes that — only more data does. `summarize()` reports
`sharpe_std_error` and `sharpe_t_stat` for exactly this reason; below t ≈ 2,
the headline number means nothing. This bit us directly: an early test
asserted `sharpe < 2.5` on 1500 bars and failed on 2 of 5 random-walk seeds,
because it was testing noise.

**Walk-forward, never a single split.** Fit on `[t, t+train)`, evaluate on
`[t+train, t+train+test)`, step, repeat. One split yields one out-of-sample
number; when it disappoints you will "fix" the strategy and look again, and
the test set is now contaminated. Per-fold **parameter churn** is as
informative as the returns — parameters that jump between folds are fitting
noise even when the stitched curve looks fine (`WalkForwardResult.param_stability`).

**Count your trials honestly.** `deflated_sharpe_ratio` *requires*
`n_trials`, because the right benchmark is not zero — it is the expected
maximum Sharpe of `n_trials` worthless strategies. Same observed Sharpe of
0.05 per bar:

| Trials | Deflated Sharpe |
|---|---|
| 1 | 0.94 |
| 10 | 0.72 |
| 50 | 0.56 |
| 500 | 0.36 |

Nothing about the strategy changed. Only how hard you looked. Log every
variant in `RESEARCH_LOG.md`, including abandoned ones — understating
`n_trials` is the easiest way to fool yourself here.

**PBO (`validation/pbo.py`).** CSCV asks: when you pick the in-sample best,
how often is it below median out-of-sample? Near 0.5 means your *selection
procedure* has no skill, and the honest response is to stop tuning rather than
tune differently. It measures the procedure, so feed it the whole family,
rejects included.

**Run the benchmark unconstrained.** Under the configured limits,
buy-and-hold is not a benchmark: a 0.30 per-symbol cap forces continuous
selling of whichever asset is winning, which turned a 4-trade hold into
**6,754 trades** and a capped rebalancing strategy. The CLI therefore runs the
reference with no per-symbol cap and no halt. The useful consequence is that
the gap between a constrained strategy and an unconstrained hold now includes
**the cost of the risk policy itself** — a real cost, better visible than
netted away.

**Beat buy-and-hold, after costs.** Report `BuyAndHold` alongside every
result. Over most crypto windows a strategy that fails to beat it has produced
nothing but turnover. Note that holding is *not* the same as requesting equal
weights each bar — that is `EqualWeightRebalanced`, a different and more
expensive strategy. On trend-generated data they differ sharply (+31% versus
−79%).

**Measure before optimising, and pin the output first.** The engine was ~10x
slower than it needed to be, and the written diagnosis here was wrong about
why: it blamed per-bar `.iloc` access in the engine's own loop. A profiler said
otherwise — **57% of total runtime was inside a single strategy method**,
`CrossSectionalMomentum._held`, which recomputes trailing returns on every
non-rebalance bar (deliberately, see §3) and was doing it with pandas
`sort_values`/`dropna` on four numbers. Each pandas row object cost ~48 µs; the
arithmetic it wrapped cost nanoseconds.

Two things followed. First, the fix was mostly in the *strategies*, not the
engine, which no amount of reading the code had suggested. Second, and more
important: optimising a backtest engine is unusually dangerous, because a
change that is 10x faster and 0.3% different silently invalidates every result
recorded before it, and nothing in a returns series announces that it moved.
So the output was pinned *first* — `tests/test_engine_golden.py` hashes the
exact float repr of every equity point, fill and violation across seven cases
spanning data regimes, strategy families, a halting run and a turnover-clamped
run. Every step of the optimisation was then required to reproduce those
digests exactly, which it does: the speedup changed no result. Threshold
assertions ("Sharpe above 1") would not have caught a drift of this kind.

**Pin your data.** Exchange history is not immutable; candles get revised and
APIs backfill. One published project saw in-sample CAGR move from −0.7% to
+1.8% between two downloads of nominally the same dataset. Snapshots are
content-addressed, manifests are committed, and `load()` re-verifies the hash.

---

## 10. Deployment

Target is a Hetzner box: Docker + Compose, restart on failure.

```bash
docker compose run --rm fetch         # pin a data snapshot (needs exchange access)
docker compose run --rm backtest      # measure
docker compose run --rm paper-replay  # dry-run the daemon against pinned data
docker compose --profile paper up -d  # long-running paper trader
```

The `paper` service restarts `unless-stopped`, which is why the halt sentinel
has to be sticky (section 8), and its healthcheck is `--watchdog`, which is
why the heartbeat has to be written from inside and read from outside
(section 6). Run `paper-replay` before trusting a deployment: it exercises the
journal, the startup reconciliation and the gate at CPU speed rather than one
bar an hour.

Three notes from building this:

- **Exchange APIs must be reachable from wherever this runs.** Claude's cloud
  sandbox blocks them by default, so snapshots were built from synthetic data
  here; the first real fetch has to happen on the host.
- **The engine is a per-bar Python loop, now with the pandas taken out of the
  hot path.** It runs at ~0.11 ms/bar: a 51k-bar (2021→now, hourly) backtest
  takes ~5.5s and a 32-fold × 6-combo walk-forward about a minute. That is
  ~10x faster than it was, and the change is pinned to produce bit-identical
  results — see §9 and `tests/test_engine_golden.py`.
- **State lives in a bind mount, not the image.** `state/` holds the journal,
  the book checkpoint and the halt sentinel, so a rebuild or redeploy must not
  lose it: the daemon's whole recovery story is reading that directory back.

---

## 11. Known limitations

Honest list, in rough priority order:

1. **Nothing has been measured on real data.** Exchange APIs are unreachable
   from the sandbox this was built in, so every result so far is from
   synthetic data and proves plumbing, not profitability. Pinning a real
   snapshot on the host is the next piece of work, and until it happens there
   are no findings here at all.
2. **Alerting does not reach a human.** The heartbeat and `--watchdog` exist
   and the healthcheck goes unhealthy, but nothing pages anybody. Item 5 of
   the section 7 gate is therefore still open.
3. **Engine performance is adequate, not optimal.** ~0.11 ms/bar after the
   numpy work (§9); a full walk-forward is about a minute. The remaining easy
   win is that a walk-forward is embarrassingly parallel across
   (fold × parameter combination) and currently runs on one core, which is
   worth doing when a grid gets large enough to be annoying and not before.
4. **No slippage model from real order books.** The square-root impact law is
   a reasonable prior, not a measurement. Paper trading against live quotes is
   what will calibrate it — see the volume-estimation note in section 8 for
   where the divergence will first show up.
5. **Paper reconciliation has no counterparty.** It verifies our journal
   against our own checkpoint, which is an integrity check, not a venue
   check. See section 6.
6. **Single-venue, spot, long-only.** No margin, no shorting, no funding
   rates, no cross-venue execution. All deliberate for a first system.
7. **No regime labelling.** Results are reported over whole samples; splitting
   by volatility regime would be more informative.
8. **Synthetic-data results prove plumbing, not profitability.** Momentum
   scoring 14x on trend-generated data is a tautology, not a finding.
