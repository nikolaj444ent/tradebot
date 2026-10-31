"""Append-only run state: the journal, the book checkpoint, the heartbeat.

A backtest can keep everything in memory because it starts and ends inside one
process. A paper run cannot: it is expected to outlive restarts, deploys and
crashes, and the one failure that actually costs money is the process dying
silently while holding a position (DESIGN.md section 6). So every decision,
order and fill is written down before it matters, and the daemon rebuilds its
world from that record on startup rather than starting flat and trading blind.

Three artefacts, each with a different job:

  journal.jsonl   Append-only, never rewritten. The authority on what
                  happened. Replaying it reconstructs the portfolio, the
                  idempotency keys already used, and the equity marks the
                  risk gate's stops key off.
  book.json       A checkpoint of cash and positions, rewritten each bar.
                  Not the authority — the journal is — which is exactly what
                  makes it useful: if the two disagree, state is corrupt and
                  the daemon must halt instead of guessing.
  heartbeat.json  Proof of life, with the bar it last processed. Something
                  outside this process has to read it; see `heartbeat_age`
                  and `python -m tradebot.run.paper --watchdog`.

Why JSONL and not a database: the log is small (one handful of lines per bar),
it has to survive being inspected by a human at 3am with `tail`, and a
corrupted trailing line must cost at most one bar. A half-written SQLite page
costs everything.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .types import Fill, Order, RiskViolation, Side

if TYPE_CHECKING:  # pragma: no cover - import-time layering, see below
    from .engine.portfolio import Portfolio

SCHEMA_VERSION = 1

#: Written when a risk limit halts the run. Its presence blocks a restart.
#:
#: This file is what makes the circuit breaker real. `docker compose` restarts
#: the paper service `unless-stopped`, so without a sticky marker a halt would
#: be followed seconds later by a fresh process resuming trading — a breaker
#: that trips and immediately re-closes, which is decoration. Clearing it is a
#: human decision, which is the whole point of a halt.
HALT_SENTINEL = "HALTED"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True, slots=True)
class RebuiltState:
    """What a restart recovers from the journal.

    `marks` is in bar order and gets replayed through a fresh `RiskGate` so
    that the drawdown stop keeps its high-water mark across restarts. Without
    that, every restart would reset the gate's memory and a process that
    crash-loops would never breach a drawdown limit however far equity fell.
    """

    portfolio: "Portfolio"
    marks: list[tuple[pd.Timestamp, float]]
    filled_order_ids: set[str]
    last_bar_ts: pd.Timestamp | None
    n_events: int
    run_id: str | None
    initial_equity: float | None


class Journal:
    """Append-only event log for one paper run, plus its checkpoints.

    One directory per run name, so two strategies can paper-trade side by side
    without sharing state. Opening a journal never truncates anything: the log
    is only ever appended to, because the day you need it is the day after
    something went wrong.
    """

    def __init__(self, root: str | Path, run_name: str = "paper") -> None:
        self.dir = Path(root) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.path = self.dir / "journal.jsonl"
        self.book_path = self.dir / "book.json"
        self.heartbeat_path = self.dir / "heartbeat.json"
        self.halt_path = self.dir / HALT_SENTINEL
        self._seq = self._last_seq()

    # ---------------------------------------------------------------- writing

    def _last_seq(self) -> int:
        seq = 0
        for event in self.replay():
            seq = max(seq, int(event.get("seq", 0)))
        return seq

    def append(self, event: str, **fields) -> dict:
        """Write one event and flush it to disk before returning.

        The fsync is deliberate. An order that exists in the exchange's memory
        but not in ours is the single worst state this system can be in, so the
        record of intent goes to stable storage before the order is placed —
        not after, and not whenever the OS feels like it.
        """
        self._seq += 1
        record = {
            "seq": self._seq,
            "schema_version": SCHEMA_VERSION,
            "wall_ts": _utcnow().isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def log_order(self, order: Order) -> None:
        self.append(
            "order",
            client_order_id=order.client_order_id,
            ts_decided=_ts(order.ts_decided),
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            reason=order.reason,
        )

    def log_fill(self, fill: Fill) -> None:
        self.append(
            "fill",
            client_order_id=fill.client_order_id,
            ts_filled=_ts(fill.ts_filled),
            symbol=fill.symbol,
            side=fill.side.value,
            qty=fill.qty,
            price=fill.price,
            fee=fill.fee,
            slippage_bps=fill.slippage_bps,
        )

    def log_violation(self, violation: RiskViolation) -> None:
        self.append(
            "violation",
            ts=_ts(violation.ts),
            rule=violation.rule,
            detail=violation.detail,
            symbol=violation.symbol,
        )

    def log_mark(self, ts: pd.Timestamp, equity: float, weights: dict[str, float]) -> None:
        """Record mark-to-market equity for one bar.

        This is the paper run's equity curve, and it is written one bar at a
        time rather than assembled at the end, because a run that ends by being
        killed still has to leave a readable curve behind.
        """
        self.append("mark", ts=_ts(ts), equity=float(equity), weights=weights)

    # ---------------------------------------------------------------- reading

    def replay(self):
        """Yield every event in order, skipping a corrupt trailing line.

        A line that fails to parse is almost always the last one, half-written
        by a process that died mid-append. Tolerating exactly that case costs
        us one bar of history; refusing to parse the file at all would cost the
        entire run. A corrupt line anywhere else raises, because that is
        damage, not a crash.
        """
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if i == len(lines) - 1:
                    return  # torn final write: the process died mid-append
                raise ValueError(
                    f"{self.path}:{i + 1} is corrupt and is not the last line. "
                    f"The journal is the authority on what happened, so this is "
                    f"not safe to skip: {exc}"
                ) from exc

    def rebuild(self, initial_equity: float) -> RebuiltState:
        """Reconstruct the portfolio by replaying fills.

        Note what this does *not* do: trust any stored position total. Cash and
        positions are derived by applying every recorded fill in order, so the
        reconstruction is wrong only if the fills themselves are wrong.
        """
        # Imported here, not at module scope, and not for style: this module is
        # the persistence layer and `engine` sits above it, so importing the
        # engine package at import time makes `import tradebot.state` a cycle
        # (engine/__init__ pulls in engine.paper, which needs this module).
        from .engine.portfolio import Portfolio

        pf = Portfolio(cash=initial_equity)
        marks: list[tuple[pd.Timestamp, float]] = []
        seen: set[str] = set()
        last_bar: pd.Timestamp | None = None
        n = 0
        run_id = None
        logged_initial: float | None = None

        for event in self.replay():
            n += 1
            kind = event.get("event")
            if kind == "run_started":
                run_id = event.get("run_id", run_id)
                if logged_initial is None and event.get("initial_equity") is not None:
                    logged_initial = float(event["initial_equity"])
            elif kind == "fill":
                fill = Fill(
                    client_order_id=event["client_order_id"],
                    ts_filled=pd.Timestamp(event["ts_filled"]),
                    symbol=event["symbol"],
                    side=Side(event["side"]),
                    qty=float(event["qty"]),
                    price=float(event["price"]),
                    fee=float(event["fee"]),
                    slippage_bps=float(event.get("slippage_bps", 0.0)),
                )
                if fill.client_order_id in seen:
                    # The log should never contain two fills for one id; if it
                    # does, applying both would double the position.
                    raise ValueError(
                        f"journal contains a duplicate fill for "
                        f"{fill.client_order_id}. Idempotency has already been "
                        f"violated; do not trade on this state."
                    )
                seen.add(fill.client_order_id)
                pf.apply_fill(fill)
            elif kind == "mark":
                ts = pd.Timestamp(event["ts"])
                marks.append((ts, float(event["equity"])))
                last_bar = ts

        # A journal started with different capital describes a different run.
        # Silently reusing it would make the equity curve a fiction.
        if logged_initial is not None and abs(logged_initial - initial_equity) > 1e-6:
            raise ValueError(
                f"journal {self.path} was started with initial_equity "
                f"{logged_initial:.2f} but config says {initial_equity:.2f}. "
                f"Use a different --run-name rather than mixing two runs in one "
                f"equity curve."
            )

        return RebuiltState(
            portfolio=pf,
            marks=marks,
            filled_order_ids=seen,
            last_bar_ts=last_bar,
            n_events=n,
            run_id=run_id,
            initial_equity=logged_initial,
        )

    def equity_curve(self) -> pd.Series:
        """The recorded equity marks, as a series indexed by bar close."""
        ts: list[pd.Timestamp] = []
        vals: list[float] = []
        for event in self.replay():
            if event.get("event") == "mark":
                ts.append(pd.Timestamp(event["ts"]))
                vals.append(float(event["equity"]))
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="equity")

    def fills(self) -> list[Fill]:
        out = []
        for event in self.replay():
            if event.get("event") == "fill":
                out.append(
                    Fill(
                        client_order_id=event["client_order_id"],
                        ts_filled=pd.Timestamp(event["ts_filled"]),
                        symbol=event["symbol"],
                        side=Side(event["side"]),
                        qty=float(event["qty"]),
                        price=float(event["price"]),
                        fee=float(event["fee"]),
                        slippage_bps=float(event.get("slippage_bps", 0.0)),
                    )
                )
        return out

    def violations(self) -> list[RiskViolation]:
        out = []
        for event in self.replay():
            if event.get("event") == "violation":
                out.append(
                    RiskViolation(
                        ts=pd.Timestamp(event["ts"]),
                        rule=event["rule"],
                        detail=event.get("detail", ""),
                        symbol=event.get("symbol"),
                    )
                )
        return out

    # ------------------------------------------------------------ checkpoints

    def write_book(self, pf: "Portfolio", ts, equity: float) -> None:
        """Checkpoint cash and positions, written after the journal.

        Order matters: the journal is appended first, the book second. If the
        process dies between the two, the book is one bar stale and
        `reconcile` catches it. If it were written first, a crash would leave a
        book claiming positions no fill accounts for — which is precisely the
        state that cannot be told apart from a lost fill.
        """
        payload = {
            "seq": self._seq,
            "ts": _ts(ts),
            "wall_ts": _utcnow().isoformat(),
            "cash": pf.cash,
            "positions": dict(pf.positions),
            "fees_paid": pf.fees_paid,
            "n_fills": pf.n_fills,
            "equity": float(equity),
        }
        tmp = self.book_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        os.replace(tmp, self.book_path)  # atomic: never a half-written book

    def read_book(self) -> dict | None:
        if not self.book_path.exists():
            return None
        try:
            return json.loads(self.book_path.read_text())
        except json.JSONDecodeError:
            return None

    def write_heartbeat(self, *, bar_ts, equity: float, interval_s: float, note: str = "") -> None:
        payload = {
            "wall_ts": _utcnow().isoformat(),
            "monotonic": time.monotonic(),
            "bar_ts": _ts(bar_ts),
            "equity": float(equity),
            "pid": os.getpid(),
            "interval_s": interval_s,
            "note": note,
        }
        tmp = self.heartbeat_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        os.replace(tmp, self.heartbeat_path)

    def heartbeat_age(self) -> float | None:
        """Seconds since the last heartbeat, or None if there is none.

        Read by the watchdog, not by the daemon. A process cannot detect its
        own death, which is why this number has to be interpreted from
        outside.
        """
        hb = self.heartbeat_path
        if not hb.exists():
            return None
        try:
            payload = json.loads(hb.read_text())
            written = datetime.fromisoformat(payload["wall_ts"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        return (_utcnow() - written).total_seconds()

    # ------------------------------------------------------------------ halts

    def halt(self, reason: str, *, ts=None, positions: dict | None = None) -> None:
        self.append("halt", reason=reason, ts=_ts(ts) if ts is not None else None)
        self.halt_path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "ts": _ts(ts) if ts is not None else None,
                    "wall_ts": _utcnow().isoformat(),
                    "positions_at_halt": positions or {},
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

    def halted(self) -> dict | None:
        if not self.halt_path.exists():
            return None
        try:
            return json.loads(self.halt_path.read_text())
        except json.JSONDecodeError:
            return {"reason": "unreadable halt sentinel"}

    def clear_halt(self) -> bool:
        """Remove the sentinel. Only ever called by an explicit human command."""
        if self.halt_path.exists():
            self.halt_path.unlink()
            self.append("halt_cleared")
            return True
        return False


@dataclass(frozen=True, slots=True)
class Reconciliation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.ok:
            return "state reconciles"
        return "state does NOT reconcile:\n  - " + "\n  - ".join(self.problems)


def reconcile(
    rebuilt: "Portfolio",
    venue_positions: dict[str, float] | None,
    *,
    book: dict | None = None,
    qty_tolerance: float = 1e-8,
    cash_tolerance: float = 1e-4,
) -> Reconciliation:
    """Compare the journal-derived book against the venue and the checkpoint.

    DESIGN.md section 6 requires a paper or live run to check its own state on
    startup and halt on mismatch rather than trade blind. Two comparisons, and
    they are not equally strong:

    `venue_positions` is the real check — someone else's record of what we own.
    In paper mode there is no someone else: fills are simulated locally, so the
    caller passes None and this degrades to an integrity check of our own
    files. That is worth doing (it catches a truncated journal, a hand-edited
    book, a crash between the two writes) but it is emphatically *not* the same
    guarantee, and a live adapter must pass real venue positions here rather
    than inherit the weaker check by default.
    """
    problems: list[str] = []
    detail: dict = {}

    if book is not None:
        book_pos = {k: float(v) for k, v in (book.get("positions") or {}).items()}
        for sym in sorted(set(book_pos) | set(rebuilt.positions)):
            want = book_pos.get(sym, 0.0)
            have = rebuilt.position_qty(sym)
            if abs(want - have) > qty_tolerance:
                problems.append(
                    f"{sym}: checkpoint says {want:.10g}, replaying fills gives "
                    f"{have:.10g}"
                )
        book_cash = book.get("cash")
        if book_cash is not None and abs(float(book_cash) - rebuilt.cash) > cash_tolerance:
            problems.append(
                f"cash: checkpoint says {float(book_cash):.6f}, replaying fills "
                f"gives {rebuilt.cash:.6f}"
            )
        detail["book"] = book_pos

    if venue_positions is not None:
        detail["venue"] = dict(venue_positions)
        for sym in sorted(set(venue_positions) | set(rebuilt.positions)):
            want = float(venue_positions.get(sym, 0.0))
            have = rebuilt.position_qty(sym)
            if abs(want - have) > qty_tolerance:
                problems.append(
                    f"{sym}: venue holds {want:.10g}, our books say {have:.10g}"
                )

    detail["rebuilt"] = dict(rebuilt.positions)
    detail["venue_checked"] = venue_positions is not None
    return Reconciliation(ok=not problems, problems=problems, detail=detail)
