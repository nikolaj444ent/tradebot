"""The paper-trading daemon.

The long-running counterpart of ``engine/backtest.py``, built around one
invariant: **the decision path is the backtest's decision path.** Same
strategy object, same risk gate, same portfolio accounting, same order sizing,
same fill model. The only thing that changes is where bars and execution
prices come from (``MarketFeed``). ``tests/test_paper.py`` holds the proof:
driven over a replayed panel, the daemon reproduces ``run_backtest`` fill for
fill and equity mark for equity mark.

Per bar, the sequence mirrors the backtest loop:

    bar i closes -> mark equity (prices the fills from bar i-1's decision)
                 -> risk gate: observe, halt?
                 -> strategy decides from history <= i   (never sees i+1)
                 -> gate clamps
                 -> orders fill NOW at the live reference (~ bar i+1's open)

What the daemon adds is everything a crash can break (DESIGN.md section 6):

- every event is journaled before the book moves on;
- ``client_order_id`` is deterministic, and the set of filled ids is reloaded
  from the journal on startup, so a replayed bar cannot double-fill;
- startup reconciles the journal against the snapshot and refuses to trade on
  a mismatch;
- a halt flattens the book and is persistent until a human clears it —
  Docker's restart policy cannot quietly resume a halted run;
- the risk gate's memory (high-water mark, day-open equity) survives
  restarts, so a crash cannot reset the drawdown stop;
- a heartbeat file lets a watchdog tell a quiet daemon from a dead one.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import pandas as pd

from ..costs import CostModel
from ..engine.portfolio import Portfolio
from ..execution.base import ExecutionContext
from ..execution.sim import SimExecution
from ..risk import RiskGate, RiskLimits
from ..strategy.base import Strategy
from ..types import Fill, MarketView, OHLCV_FIELDS
from .alerts import send_alert
from .feed import MarketFeed
from .journal import (
    Journal,
    PaperState,
    fill_from_event,
    fill_to_event,
    load_snapshot,
    write_snapshot,
)


class PaperHalted(RuntimeError):
    """The risk gate stopped the run. Restarting it is a human decision."""


class ReconciliationError(RuntimeError):
    """Journal and snapshot disagree. Do not trade a book you cannot vouch for."""


class PaperTrader:
    def __init__(
        self,
        *,
        feed: MarketFeed,
        strategy: Strategy,
        costs: CostModel,
        limits: RiskLimits,
        state: PaperState,
        initial_equity: float = 10_000.0,
        cash_buffer: float = 0.02,
        run_id: str | None = None,
        alert: Callable[[str, str], None] = send_alert,
    ):
        if not limits.halt_on_breach:
            raise ValueError(
                "paper trading requires halt_on_breach=True. A circuit breaker "
                "that does not break the circuit is decoration — see DESIGN.md "
                "section 5. halt_on_breach=False is for research backtests only."
            )
        self.feed = feed
        self.strategy = strategy
        self.costs = costs
        self.limits = limits
        self.state = state
        self.initial_equity = float(initial_equity)
        self.cash_buffer = cash_buffer
        self._requested_run_id = run_id
        self.alert = alert

        self.journal = Journal(state.journal_path)
        self.run_id: str = ""
        self.origin: pd.Timestamp | None = None
        self.pf = Portfolio(cash=self.initial_equity)
        self.gate = RiskGate(limits=limits)
        self.adapter: SimExecution | None = None
        self.fills: list[Fill] = []
        self._marks: list[tuple[pd.Timestamp, float]] = []
        self._last_decided: pd.Timestamp | None = None
        self._started = False

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        reason = self.state.halted_reason()
        if reason is not None:
            raise PaperHalted(
                f"this run is HALTED: {reason.strip()}\n"
                f"A halted run does not quietly resume (DESIGN.md section 5). "
                f"Decide deliberately, then clear it with --clear-halt."
            )
        self.state.ensure()
        self.state.heartbeat()

        events = self.journal.read()
        warmup = max(int(self.strategy.warmup), 1)
        if events:
            self._resume(events, warmup)
        else:
            self._fresh_start(warmup)
        self._started = True
        self._on_panel(self._bootstrap_panel)
        del self._bootstrap_panel

    def _fresh_start(self, warmup: int) -> None:
        panel = self.feed.bootstrap(warmup)
        self.origin = panel["close"].index[0]
        self.run_id = self._requested_run_id or uuid.uuid4().hex[:12]
        self.adapter = SimExecution(costs=self.costs, mode="paper")
        self.journal.append(
            {
                "type": "started",
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "run_id": self.run_id,
                "origin": self.origin.isoformat(),
                "strategy": self.strategy.describe(),
                "initial_equity": self.initial_equity,
                "timeframe": self.feed.timeframe,
            }
        )
        self._write_snapshot()
        self._bootstrap_panel = panel

    def _resume(self, events: list[dict], warmup: int) -> None:
        started = next((e for e in events if e.get("type") == "started"), None)
        if started is None:
            self._fail_reconcile("journal has events but no 'started' record")
        if started["strategy"] != self.strategy.describe():
            raise ValueError(
                f"this run was started with {started['strategy']} but the daemon "
                f"was given {self.strategy.describe()}. Changing the strategy "
                f"mid-run invalidates the book; start a new --run-name instead."
            )
        if abs(started["initial_equity"] - self.initial_equity) > 1e-9:
            raise ValueError(
                f"initial_equity changed from {started['initial_equity']} to "
                f"{self.initial_equity} mid-run. Start a new --run-name instead."
            )

        fills = [fill_from_event(e) for e in events if e.get("type") == "fill"]
        replayed = Portfolio(cash=self.initial_equity)
        for f in fills:
            replayed.apply_fill(f)

        snap = load_snapshot(self.state.snapshot_path)
        if snap is None:
            self._fail_reconcile(
                "journal exists but snapshot.json is missing. The snapshot is "
                "written before the first decision, so its absence means state "
                "was deleted or the directory is damaged."
            )
        if snap["run_id"] != started["run_id"]:
            self._fail_reconcile("snapshot and journal belong to different runs")
        n = int(snap["n_fills"])
        if n > len(fills):
            self._fail_reconcile(
                f"snapshot claims {n} fills but the journal holds {len(fills)}. "
                f"The journal is append-only truth; it can run ahead of the "
                f"snapshot after a crash, never behind it."
            )
        partial = Portfolio(cash=self.initial_equity)
        for f in fills[:n]:
            partial.apply_fill(f)
        if abs(partial.cash - snap["cash"]) > 1e-6:
            self._fail_reconcile(
                f"cash mismatch: journal replay says {partial.cash:.8f}, "
                f"snapshot says {snap['cash']:.8f}"
            )
        for sym in set(partial.positions) | set(snap["positions"]):
            a = partial.positions.get(sym, 0.0)
            b = snap["positions"].get(sym, 0.0)
            if abs(a - b) > 1e-9:
                self._fail_reconcile(
                    f"position mismatch in {sym}: journal replay says {a}, "
                    f"snapshot says {b}"
                )

        # The journal is the authority; the snapshot may be a crash behind.
        self.pf = replayed
        self.run_id = started["run_id"]
        self.origin = pd.Timestamp(started["origin"])
        g = snap.get("gate", {})
        self.gate._high_water = float(g.get("high_water", 0.0))
        self.gate._day = pd.Timestamp(g["day"]) if g.get("day") else None
        self.gate._day_open_equity = g.get("day_open_equity")
        self.adapter = SimExecution(
            costs=self.costs, mode="paper", _seen={f.client_order_id for f in fills}
        )
        last_bar = pd.Timestamp(snap["last_bar"]) if snap.get("last_bar") else None
        self._last_decided = last_bar
        self.journal.append(
            {
                "type": "resumed",
                "ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "n_fills_replayed": len(fills),
                "cash": self.pf.cash,
            }
        )
        self._bootstrap_panel = self.feed.bootstrap(warmup, origin=self.origin, upto=last_bar)

    def _fail_reconcile(self, detail: str) -> None:
        msg = f"reconciliation failed: {detail}"
        self.state.write_halted(msg)
        try:
            self.journal.append(
                {"type": "halt", "ts": pd.Timestamp.now(tz="UTC").isoformat(), "reason": msg}
            )
        finally:
            self.alert("tradebot paper run halted on startup", msg)
        raise ReconciliationError(msg)

    # ------------------------------------------------------------------ loop

    def run_forever(
        self,
        *,
        poll_seconds: float = 30.0,
        max_steps: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Process bars until the feed is exhausted (replay) or forever (live)."""
        if not self._started:
            self.start()
        steps = 0
        while True:
            self.state.heartbeat(last_bar=self._last_decided)
            panel = self.feed.next_bar()
            if panel is None:
                if self.feed.exhausted:
                    return
                sleep(poll_seconds)
                continue
            self._on_panel(panel)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return

    def step(self) -> bool:
        """Process at most one new bar. False once a replay feed is exhausted."""
        if not self._started:
            self.start()
        panel = self.feed.next_bar()
        if panel is None:
            return not self.feed.exhausted
        self._on_panel(panel)
        return True

    # ------------------------------------------------------------- one bar

    def _on_panel(self, frames: dict[str, pd.DataFrame]) -> None:
        close = frames["close"]
        ts = close.index[-1]
        if self._last_decided is not None and ts <= self._last_decided:
            # Resumed exactly at the bar we last decided on: nothing new.
            self.state.heartbeat(last_bar=self._last_decided)
            return
        px = close.iloc[-1]

        equity = self.pf.equity(px)
        self.gate.observe_equity(ts, equity)
        if self._last_decided is not None:
            self._marks.append((ts, equity))
            self.journal.append(
                {
                    "type": "mark",
                    "ts": ts.isoformat(),
                    "equity": float(equity),
                    "cash": float(self.pf.cash),
                    "n_positions": len(self.pf.positions),
                }
            )

        ref = self.feed.execution_ref()

        reason = self.gate.halt_reason(ts, equity)
        if reason is not None:
            self._halt(ts, reason, ref)

        if ref is None:
            # Nothing can fill (replay exhausted, or the ticker is unreachable).
            # Skipping the decision is the honest move: an order with no price
            # would be a fiction. The bar is still consumed.
            if not self.feed.exhausted:
                self.journal.append(
                    {"type": "no_execution_ref", "ts": ts.isoformat()}
                )
            self._last_decided = ts
            self._write_snapshot()
            self.state.heartbeat(last_bar=ts)
            return

        view = MarketView(frames={f: frames[f] for f in OHLCV_FIELDS})
        raw = self.strategy.target_weights(view)
        targets = self.gate.clamp(ts, raw, self.pf.weights(px))
        self.journal.append(
            {
                "type": "decision",
                "ts": ts.isoformat(),
                "raw": {k: float(v) for k, v in raw.items()},
                "targets": {k: float(v) for k, v in targets.items()},
                "equity": float(equity),
            }
        )
        orders = self.pf.orders_to_reach(
            targets,
            px,
            ts=ts,
            run_id=self.run_id,
            min_notional=self.costs.min_notional,
            taker_fee_bps=self.costs.taker_fee_bps,
            cash_buffer=self.cash_buffer,
            reason=self.strategy.name,
        )
        for o in orders:
            self.journal.append(
                {
                    "type": "order",
                    "id": o.client_order_id,
                    "ts": o.ts_decided.isoformat(),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.qty),
                }
            )
        self._execute(orders, ref)
        self._last_decided = ts
        self._write_snapshot()
        self.state.heartbeat(last_bar=ts)

    def _execute(self, orders, ref) -> None:
        ctx = ExecutionContext(
            ts=ref.ts,
            ref_price=ref.ref_price,
            bar_volume_notional=ref.bar_volume_notional,
            available_cash=self.pf.cash,
        )
        for f in self.adapter.execute(orders, ctx):
            self.journal.append(fill_to_event(f))
            self.pf.apply_fill(f)
            self.fills.append(f)

    def _halt(self, ts: pd.Timestamp, reason: str, ref) -> None:
        """Flatten, persist the halt, reach a human, stop. Mirrors the backtest."""
        if ref is not None:
            orders = self.pf.flatten_orders(
                ref.ref_price, ts=ts, run_id=self.run_id, reason=f"halt: {reason}"
            )
            self._execute(orders, ref)
        self.journal.append(
            {"type": "halt", "ts": ts.isoformat(), "reason": reason, "flattened": ref is not None}
        )
        self._write_snapshot(last_bar=ts)
        self.state.write_halted(f"{ts.isoformat()}  {reason}")
        self.alert(
            "tradebot paper run HALTED",
            f"{reason}\nBook flattened: {ref is not None}. Cash: {self.pf.cash:.2f}. "
            f"Clear with --clear-halt after deciding deliberately.",
        )
        raise PaperHalted(reason)

    # ------------------------------------------------------------ persistence

    def _write_snapshot(self, last_bar: pd.Timestamp | None = None) -> None:
        last = last_bar if last_bar is not None else self._last_decided
        write_snapshot(
            self.state.snapshot_path,
            {
                "run_id": self.run_id,
                "origin": self.origin.isoformat() if self.origin is not None else None,
                "last_bar": last.isoformat() if last is not None else None,
                "initial_equity": self.initial_equity,
                "cash": float(self.pf.cash),
                "positions": {k: float(v) for k, v in self.pf.positions.items()},
                "fees_paid": float(self.pf.fees_paid),
                "n_fills": int(self.pf.n_fills),
                "gate": {
                    "high_water": float(self.gate._high_water),
                    "day": self.gate._day.isoformat() if self.gate._day is not None else None,
                    "day_open_equity": self.gate._day_open_equity,
                },
                "strategy": self.strategy.describe(),
            },
        )

    # -------------------------------------------------------------- reporting

    @property
    def equity_series(self) -> pd.Series:
        if not self._marks:
            return pd.Series(dtype="float64", name="equity")
        ts, vals = zip(*self._marks)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="equity")
