"""Read-only views over paper-run state, for the dashboard.

Everything here is derived from the journal — the append-only truth — plus
the heartbeat file's mtime. Nothing in this module writes. The daemon owns
the state directory; the dashboard only looks at it, so a dashboard bug can
never corrupt a run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .journal import PaperState


def _max_drawdown(equity: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, 1.0 - v / peak)
    return worst


def run_report(state: PaperState) -> dict | None:
    """One run's journal, folded into what a reader needs. None if no run."""
    if not state.journal_path.exists():
        return None
    events: list[dict] = []
    lines = state.journal_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn final line after a crash; same rule as Journal.read
            raise

    started = next((e for e in events if e["type"] == "started"), None)
    if started is None:
        return None

    marks = [e for e in events if e["type"] == "mark"]
    fills = [e for e in events if e["type"] == "fill"]
    decisions = [e for e in events if e["type"] == "decision"]
    halts = [e for e in events if e["type"] == "halt"]

    snap = {}
    if state.snapshot_path.exists():
        snap = json.loads(state.snapshot_path.read_text())

    heartbeat_age = None
    if state.heartbeat_path.exists():
        heartbeat_age = time.time() - state.heartbeat_path.stat().st_mtime

    initial = float(started["initial_equity"])
    equity_curve = [[e["ts"], e["equity"]] for e in marks]
    last_equity = marks[-1]["equity"] if marks else initial

    return {
        "name": state.root.name,
        "run_id": started["run_id"],
        "strategy": started["strategy"],
        "timeframe": started.get("timeframe", "?"),
        "started": started["ts"],
        "origin": started["origin"],
        "last_bar": snap.get("last_bar"),
        "initial_equity": initial,
        "equity": last_equity,
        "pnl_pct": (last_equity / initial - 1.0) if initial else 0.0,
        "max_drawdown": _max_drawdown([initial] + [m["equity"] for m in marks]),
        "cash": snap.get("cash"),
        "positions": snap.get("positions", {}),
        "fees_paid": snap.get("fees_paid", sum(f["fee"] for f in fills)),
        "n_fills": len(fills),
        "n_decisions": len(decisions),
        "n_marks": len(marks),
        "heartbeat_age_s": heartbeat_age,
        "halted": state.halted_reason(),
        "halts": [{"ts": h["ts"], "reason": h["reason"]} for h in halts],
        "equity_curve": equity_curve,
        # Newest first; the page shows these in a table.
        "fills": list(reversed(fills)),
        "last_decision": decisions[-1] if decisions else None,
    }


def all_runs(state_dir: str | Path) -> list[dict]:
    base = Path(state_dir) / "paper"
    if not base.exists():
        return []
    reports = []
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        r = run_report(PaperState(root=run_dir))
        if r is not None:
            reports.append(r)
    return reports
