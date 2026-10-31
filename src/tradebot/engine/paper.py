"""The paper-trading loop.

This is the backtest loop with the clock turned the right way round, and that
framing is load-bearing: the ordering inside `on_quote` is deliberately the
same ordering as `run_backtest`, step for step.

    at the close of bar k:  mark equity at bar k's close      <- the previous
                                                                 decision's fill
                                                                 is now marked
                            observe equity in the risk gate
                            check the stops; a breach flattens and halts
                            the strategy sees history up to and including k
                            the gate clamps what it asked for
                            orders are sized at bar k's close
    immediately after:      they fill at bar k+1's OPEN, with costs

Those are the same six steps in the same order as the backtest, which is why
`tests/test_paper.py` can assert the stronger thing: replay a panel through
this loop and the equity curve equals the backtest's on that panel, bar for
bar. Parity is not a claim in a design document, it is an assertion in a test.

What this loop adds over the backtest is everything that follows from running
unattended for weeks:

* It processes each bar **once**, and knows which bars it has already seen, so
  a restart resumes rather than re-trades.
* It writes every decision, order and fill to the journal **before** acting.
* It will not start at all if state does not reconcile, if a previous run
  halted, or if the risk gate has been configured not to halt.
* A halt is sticky. It flattens, writes a sentinel, and stays down until a
  human clears it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from ..config import Config
from ..costs import CostModel
from ..data.quotes import Quote, QuoteSource, StaleFeed
from ..execution.base import ExecutionContext
from ..execution.paper import PaperExecution
from ..risk import RiskGate
from ..state import Journal, reconcile
from ..strategy.base import Strategy
from ..types import BacktestResult, MarketView
from .portfolio import Portfolio

#: Status values `on_quote` can return.
WARMING_UP = "warming_up"
ALREADY_SEEN = "already_seen"
TRADED = "traded"
HALTED = "halted"


@dataclass(frozen=True, slots=True)
class StepOutcome:
    status: str
    bar_ts: pd.Timestamp
    equity: float
    n_orders: int = 0
    n_fills: int = 0
    targets: dict[str, float] = field(default_factory=dict)
    halt_reason: str | None = None
    clamped: bool = False

    @property
    def halted(self) -> bool:
        return self.status == HALTED


class PaperStartupRefused(RuntimeError):
    """The daemon refused to start. Deliberately not recoverable in-process."""


@dataclass(slots=True)
class PaperTrader:
    """One bar at a time, with everything written down.

    Constructed by `start_paper_trader`, which is where the startup checks
    live. Building one directly skips those checks and is for tests.
    """

    strategy: Strategy
    gate: RiskGate
    portfolio: Portfolio
    adapter: PaperExecution
    journal: Journal
    costs: CostModel
    run_id: str
    cash_buffer: float = 0.02
    last_bar_ts: pd.Timestamp | None = None
    n_decisions: int = 0
    n_clamped: int = 0
    halt_reason: str | None = None
    _violation_watermark: int = 0

    # ------------------------------------------------------------------ helpers

    def _journal_new_violations(self) -> int:
        """Write any violations the gate recorded since the last call."""
        new = self.gate.violations[self._violation_watermark :]
        for violation in new:
            self.journal.log_violation(violation)
        self._violation_watermark = len(self.gate.violations)
        return len(new)

    def _flatten(self, quote: Quote, reason: str) -> list:
        orders = self.portfolio.flatten_orders(
            quote.ref_price, ts=quote.bar_ts, run_id=self.run_id, reason=reason
        )
        ctx = self._context(quote)
        fills = self.adapter.execute(orders, ctx)
        for fill in fills:
            self.portfolio.apply_fill(fill)
        return fills

    def _context(self, quote: Quote) -> ExecutionContext:
        return ExecutionContext(
            ts=quote.fill_ts,
            ref_price=quote.ref_price,
            bar_volume_notional=quote.bar_volume_notional,
            available_cash=self.portfolio.cash,
        )

    # --------------------------------------------------------------------- step

    def on_quote(self, quote: Quote) -> StepOutcome:
        """Process one closed bar. Safe to call again with the same bar."""
        if self.halt_reason is not None:
            return StepOutcome(
                status=HALTED,
                bar_ts=quote.bar_ts,
                equity=self.portfolio.equity(quote.bars["close"].iloc[-1]),
                halt_reason=self.halt_reason,
            )

        ts = quote.bar_ts
        px_decide = quote.bars["close"].iloc[-1]

        # Bars arrive at most once, but a restart can re-present the bar that
        # was in flight when the process died. Re-deciding would be *safe* —
        # client_order_id is derived from the bar, so the adapter rejects the
        # duplicate — but it would double the equity mark and corrupt the
        # curve. So the bar watermark is the authority on what is new.
        if self.last_bar_ts is not None and ts <= self.last_bar_ts:
            return StepOutcome(
                status=ALREADY_SEEN,
                bar_ts=ts,
                equity=self.portfolio.equity(px_decide),
            )

        view = MarketView(frames=dict(quote.bars))
        warmup = max(int(self.strategy.warmup), 1)
        if len(view) < warmup:
            self.journal.append(
                "warming_up", ts=str(ts), have_bars=len(view), need_bars=warmup
            )
            return StepOutcome(
                status=WARMING_UP, bar_ts=ts, equity=self.portfolio.equity(px_decide)
            )

        equity = self.portfolio.equity(px_decide)

        # The mark belongs to the PREVIOUS decision: its fills happened at this
        # bar's open, and this is their close. The backtest records exactly this
        # number at exactly this timestamp, which is why the first decision bar
        # gets no mark — there is nothing yet to mark.
        if self.last_bar_ts is not None:
            self.journal.log_mark(ts, equity, self.portfolio.weights(px_decide))

        self.gate.observe_equity(ts, equity)

        reason = self.gate.halt_reason(ts, equity)
        self._journal_new_violations()
        if reason is not None:
            # Recorded before flattening. What it was holding when the breaker
            # tripped is the thing you want to know at 3am; what it holds
            # afterwards is, by construction, nothing.
            held = dict(self.portfolio.positions)
            fills = self._flatten(quote, f"halt: {reason}")
            self.halt_reason = reason
            self.last_bar_ts = ts
            # The post-liquidation equity is recorded as an event and a
            # checkpoint, NOT as a second mark. A mark means "equity at this
            # bar's close", and this bar's close already has one — writing
            # another would put two values at one timestamp and leave the
            # equity curve with a duplicate index the backtest never produces.
            # The liquidation filled at the *next* bar's open, and that bar's
            # close is a mark this run will never live to see.
            final_equity = self.portfolio.equity(px_decide)
            self.journal.append(
                "halt_flattened",
                ts=str(ts),
                equity_after_flatten=final_equity,
                n_fills=len(fills),
                positions_before=held,
            )
            self.journal.halt(reason, ts=ts, positions=held)
            self.journal.write_book(self.portfolio, ts, final_equity)
            return StepOutcome(
                status=HALTED,
                bar_ts=ts,
                equity=final_equity,
                n_fills=len(fills),
                halt_reason=reason,
            )

        raw = self.strategy.target_weights(view)
        before = len(self.gate.violations)
        targets = self.gate.clamp(ts, raw, self.portfolio.weights(px_decide))
        clamped = len(self.gate.violations) > before
        self._journal_new_violations()

        self.n_decisions += 1
        if clamped:
            self.n_clamped += 1

        self.journal.append(
            "decision",
            ts=str(ts),
            strategy=self.strategy.describe(),
            equity=equity,
            raw_targets={k: float(v) for k, v in raw.items()},
            targets={k: float(v) for k, v in targets.items()},
            clamped=clamped,
            ref_price={k: float(v) for k, v in quote.ref_price.items()},
            volume_is_estimate=quote.volume_is_estimate,
        )

        orders = self.portfolio.orders_to_reach(
            targets,
            px_decide,
            ts=ts,
            run_id=self.run_id,
            min_notional=self.costs.min_notional,
            taker_fee_bps=self.costs.taker_fee_bps,
            cash_buffer=self.cash_buffer,
            reason=self.strategy.name,
        )
        fills = self.adapter.execute(orders, self._context(quote))
        for fill in fills:
            self.portfolio.apply_fill(fill)

        self.last_bar_ts = ts
        self.journal.write_book(self.portfolio, ts, equity)
        return StepOutcome(
            status=TRADED,
            bar_ts=ts,
            equity=equity,
            n_orders=len(orders),
            n_fills=len(fills),
            targets=targets,
            clamped=clamped,
        )


def start_paper_trader(
    cfg: Config,
    strategy: Strategy,
    journal: Journal,
    *,
    run_id: str | None = None,
    require_halt_on_breach: bool = True,
) -> PaperTrader:
    """Run the startup checks, then build a trader that has passed them.

    Every one of these refusals is a case where continuing would produce a
    plausible-looking equity curve that does not describe reality, which is the
    specific failure this whole repository is built to prevent.
    """
    halted = journal.halted()
    if halted is not None:
        raise PaperStartupRefused(
            f"this run halted previously and has not been cleared:\n"
            f"  reason: {halted.get('reason')}\n"
            f"  at:     {halted.get('ts')}\n"
            f"A halt is a human decision to resume. Inspect "
            f"{journal.dir}/journal.jsonl, then:\n"
            f"  python -m tradebot.run.paper --clear-halt --run-name "
            f"{journal.run_name}"
        )

    if require_halt_on_breach and not cfg.risk.halt_on_breach:
        raise PaperStartupRefused(
            "risk.halt_on_breach is False, which is the research escape hatch: "
            "breaches get recorded but nothing stops. A circuit breaker that "
            "does not break the circuit is decoration, so paper runs refuse to "
            "start with it off. See DESIGN.md section 5."
        )

    initial_equity = cfg.backtest.initial_equity
    rebuilt = journal.rebuild(initial_equity)

    adapter = PaperExecution(
        costs=cfg.costs,
        journal=journal,
        known_filled=frozenset(rebuilt.filled_order_ids),
    )

    # DESIGN.md section 6: compare internal state against the venue on startup
    # and halt on mismatch rather than trade blind. In paper mode there is no
    # venue to ask, so `venue_positions()` returns None and this reduces to an
    # integrity check between the journal and the book checkpoint. The report
    # says which check actually ran, because the weaker one must not be allowed
    # to read like the stronger one.
    check = reconcile(
        rebuilt.portfolio,
        adapter.venue_positions(),
        book=journal.read_book(),
    )
    if not check.ok:
        journal.halt(f"reconciliation failed: {'; '.join(check.problems)}")
        raise PaperStartupRefused(
            f"{check.describe()}\n\n"
            f"The journal is the authority on what happened, and it disagrees "
            f"with the checkpoint, so the current position is unknown. Resolve "
            f"this by hand before trading."
        )

    gate = RiskGate(limits=cfg.risk)
    # Replay the recorded marks through the fresh gate so the drawdown
    # high-water mark and the day's opening equity survive the restart. Without
    # this, a crash-looping process would reset its own stops every few seconds
    # and could never breach a drawdown limit however far equity fell.
    for ts, equity in rebuilt.marks:
        gate.observe_equity(ts, equity)

    run_id = run_id or rebuilt.run_id or uuid.uuid4().hex[:12]
    journal.append(
        "run_started",
        run_id=run_id,
        strategy=strategy.describe(),
        initial_equity=initial_equity,
        resumed=rebuilt.n_events > 0,
        resumed_from_bar=str(rebuilt.last_bar_ts) if rebuilt.last_bar_ts else None,
        positions=dict(rebuilt.portfolio.positions),
        cash=rebuilt.portfolio.cash,
        venue_check=check.detail.get("venue_checked", False),
        mode=adapter.mode,
        halt_on_breach=cfg.risk.halt_on_breach,
    )

    return PaperTrader(
        strategy=strategy,
        gate=gate,
        portfolio=rebuilt.portfolio,
        adapter=adapter,
        journal=journal,
        costs=cfg.costs,
        run_id=run_id,
        cash_buffer=cfg.backtest.cash_buffer,
        last_bar_ts=rebuilt.last_bar_ts,
        _violation_watermark=0,
    )


def run_paper_loop(
    trader: PaperTrader,
    source: QuoteSource,
    *,
    heartbeat_interval_s: float = 0.0,
    max_bars: int | None = None,
    on_step=None,
) -> StepOutcome | None:
    """Drive `trader` from `source` until it halts or the source ends.

    Returns the final outcome. A halt returns normally with
    `outcome.halted` set — the caller decides the exit code, because a halt is
    a result, not a crash.
    """
    outcome: StepOutcome | None = None
    n = 0
    try:
        for quote in source.quotes():
            outcome = trader.on_quote(quote)
            trader.journal.write_heartbeat(
                bar_ts=outcome.bar_ts,
                equity=outcome.equity,
                interval_s=heartbeat_interval_s,
                note=outcome.status,
            )
            if on_step is not None:
                on_step(outcome)
            if outcome.halted:
                return outcome
            # Bars the trader skipped as already-seen do not count: on a
            # resumed run the source replays history first, and counting that
            # would spend the whole budget before reaching a new bar.
            if outcome.status != ALREADY_SEEN:
                n += 1
            if max_bars is not None and n >= max_bars:
                return outcome
    except StaleFeed as exc:
        # The source already applied its own tolerance before raising, so by
        # the time this lands the feed has been unusable for several bars.
        # Holding a position against a feed we cannot see is exactly the
        # runaway case the breaker exists for.
        trader.journal.halt(f"stale feed: {exc}")
        trader.halt_reason = str(exc)
        raise
    finally:
        source.close()
    return outcome


def result_from_journal(
    journal: Journal, initial_equity: float, *, strategy: Strategy | None = None
) -> BacktestResult:
    """Rebuild a `BacktestResult` from a paper run's journal.

    So that a paper run is reported by the same `summarize()` as a backtest,
    with the same warnings about sample length and clamp rate. A paper curve
    read through a friendlier lens than the backtest it is being compared to
    would be worse than no comparison at all.
    """
    equity = journal.equity_curve()
    returns = equity.pct_change().fillna(0.0).rename("returns") if len(equity) else equity
    fills = journal.fills()
    violations = journal.violations()

    n_decisions = 0
    n_clamped = 0
    halted_at = None
    halt_reason = None
    run_id = None
    journal_strategy = None
    weight_rows: list[dict] = []
    weight_ts: list[pd.Timestamp] = []
    for event in journal.replay():
        kind = event.get("event")
        if kind == "decision":
            n_decisions += 1
            if event.get("clamped"):
                n_clamped += 1
        elif kind == "mark":
            weight_ts.append(pd.Timestamp(event["ts"]))
            weight_rows.append(event.get("weights") or {})
        elif kind == "halt":
            halt_reason = event.get("reason")
            halted_at = pd.Timestamp(event["ts"]) if event.get("ts") else None
        elif kind == "run_started":
            run_id = event.get("run_id", run_id)
            journal_strategy = event.get("strategy", journal_strategy)

    weights = (
        pd.DataFrame(weight_rows, index=pd.DatetimeIndex(weight_ts)).fillna(0.0)
        if weight_rows
        else pd.DataFrame()
    )
    by_rule: dict[str, int] = {}
    for violation in violations:
        by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1

    denom = max(n_decisions, 1)
    return BacktestResult(
        equity=equity,
        returns=returns,
        fills=fills,
        violations=violations,
        weights=weights,
        halted_at=halted_at,
        halt_reason=halt_reason,
        meta={
            # The journal is the authority on what actually ran. A report that
            # names the strategy the operator happened to type on the command
            # line, rather than the one that produced these fills, is exactly
            # the kind of plausible-but-wrong report this repo exists to stop.
            "strategy": journal_strategy
            or (strategy.describe() if strategy else "unknown"),
            "strategy_arg": strategy.describe() if strategy else None,
            "strategy_matches_journal": (
                journal_strategy is None
                or strategy is None
                or journal_strategy == strategy.describe()
            ),
            "params": strategy.params() if strategy else {},
            "n_params": len(strategy.params()) if strategy else 0,
            "run_id": run_id,
            "mode": "paper",
            "snapshot_id": "live",
            "initial_equity": initial_equity,
            "final_equity": float(equity.iloc[-1]) if len(equity) else initial_equity,
            "fees_paid": sum(f.fee for f in fills),
            "gross_traded_notional": sum(f.notional for f in fills),
            "cost_drag_pct_of_initial": (
                sum(f.fee for f in fills) / initial_equity if initial_equity else 0.0
            ),
            "n_decisions": n_decisions,
            "violations_by_rule": by_rule,
            "clamp_rate": n_clamped / denom,
            "violations_per_decision": (
                sum(
                    c
                    for r, c in by_rule.items()
                    if r not in ("halt", "breach_not_halted")
                )
                / denom
            ),
            "effective_bars": len(equity),
            # A paper run is never "complete": it is as long as it has been
            # running. Truncation here means it stopped early, i.e. halted.
            "truncated": halted_at is not None,
        },
    )
